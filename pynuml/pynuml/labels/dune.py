from .standard import StandardLabels

class DUNELabels(StandardLabels):
    def __init__(self,
                 gamma_threshold: float = 0.02,
                 hadron_threshold: float = 0.2):
        super(DUNELabels, self).__init__(gamma_threshold, hadron_threshold)

        self._labels = [
            'pion',
            'muon',
            'kaon',
            'hadron',
            'shower',
            'michel',
            'diffuse',
            'invisible'
        ]

    @property
    def pion(self):
        return self.index('MIP')

    @property
    def muon(self):
        return self.index('MIP')

    @property
    def kaon(self):
        return self.index('HIP')

    @property
    def hadron(self):
        return self.index('HIP')

    @property
    def delta(self):
        return self.index('MIP')