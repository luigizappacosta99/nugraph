"""NuGraph multifile data module"""
import os
from pathlib import Path

import h5py

from torch.utils.data import ConcatDataset

from . import NuGraphDataModule, NuGraphDataset

class MultiFileDataModule(NuGraphDataModule):
    """
    Basically the same as NuGraphDataModule but handles multiple files at once.
    Takes a directory instead of a single file.

    Metadata (planes, semantic_classes, gen, event_classes) is read from the
    FIRST file and assumed to be consistent across all files.
    WARNING: All files must be presplit via generate_samples.
    Train/test/validation is considered constant throught all files.
    """

    def __init__(self, directory: str | Path, batch_size: int, model=None):

        #take file list from directory
        directory = Path(directory)

        #build file list
        file_list = sorted(str(f) for f in directory.rglob("*") if f.suffix == ".h5")

        # check if files are present
        if not file_list:
            raise FileNotFoundError(f"No HDF5 files (extensions: .h5) found under '{directory}'.")
        
        print(f"[MultiFileDataModule] Found {len(file_list)} file(s) in '{directory}':")
        
        super().__init__(data_path=os.path.expandvars(file_list[0]), batch_size=batch_size, model=model)

        self._file_list = [os.path.expandvars(p) for p in file_list]
 
        # if there is only one file NuGraphDataModule has already done all the work
        if len(file_list) == 1:
            self._log_summary()
            return
 
        transform = model.transform(self.planes) if model else None
 
        train_datasets = [self.train_dataset]
        val_datasets   = [self.val_dataset]
        test_datasets  = [self.test_dataset]

        new_datasize = list(self.train_datasize)
 
        for filepath in self._file_list[1:]:
            self._conc_file(
                filepath,
                transform,
                train_datasets,
                val_datasets,
                test_datasets,
            )
            new_datasize = self._add_datasize(
                new_datasize, filepath
            )
        
        self.train_dataset  = ConcatDataset(train_datasets)
        self.val_dataset    = ConcatDataset(val_datasets)
        self.test_dataset   = ConcatDataset(test_datasets)
        self.train_datasize = new_datasize

        self._log_summary()

    def _conc_file(
        self,
        filepath: str,
        transform,
        train_datasets: list,
        val_datasets: list,
        test_datasets: list,
    ) -> None:
        
        with h5py.File(filepath) as f:
            self._validate_metadata(f, filepath)
 
            try:
                train_samples = f["samples/train"].asstr()[()]
                val_samples   = f["samples/validation"].asstr()[()]
                test_samples  = f["samples/test"].asstr()[()]
            except KeyError:
                print(f"[MultiFileDataModule] Sample splits not found in '{filepath}'!  Call 'generate_samples' to create them.")
 
        train_datasets.append(NuGraphDataset(filepath, train_samples, transform))
        val_datasets.append(  NuGraphDataset(filepath, val_samples,   transform))
        test_datasets.append( NuGraphDataset(filepath, test_samples,  transform))
 

    def _add_datasize(
        self,
        conc: list,
        filepath: str,
    ) -> list:
        
        with h5py.File(filepath) as f:
            try:
                extra = list(f["datasize/train"][()])
            except KeyError:
                print(f"[MultiFileDataModule] Data size array not found in '{filepath}'!  Call 'generate_samples' to create it.")
 
        return [a + b for a, b in zip(conc, extra)]
    
    def _validate_metadata(self, f: h5py.File, filepath: str) -> None:

        try:
            planes  = f["planes"].asstr()[()].tolist()
            sem_cls = f["semantic_classes"].asstr()[()].tolist()
        except KeyError:
            print(f"[MultiFileDataModule] Warning: metadata missing in '{filepath}', skipping consistency check.")
            return
 
        if planes != self.planes:
            raise ValueError(
                f"'planes' mismatch in '{filepath}':\n"
                f"  reference : {self.planes}\n"
                f"  this file : {planes}"
            )
        if sem_cls != self.semantic_classes:
            raise ValueError(
                f"'semantic_classes' mismatch in '{filepath}':\n"
                f"  reference : {self.semantic_classes}\n"
                f"  this file : {sem_cls}"
            )
 
    def _log_summary(self) -> None:
        n = len(self._file_list)
        print(
            f"[MultiFileDataModule] Ready — {n} file(s) merged:\n"
            f"  train : {len(self.train_dataset)} sample(s)\n"
            f"  val   : {len(self.val_dataset)} sample(s)\n"
            f"  test  : {len(self.test_dataset)} sample(s)"
        )
        



