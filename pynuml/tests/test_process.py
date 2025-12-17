"""Test pynuml graph processing and plotting"""
import pynuml
import nugraph as ng # pylint: disable=unused-import

def test_process_uboone():
    print("Sei sul cluster mona, che te plotti")
    """Test graph processing with MicroBooNE open data release"""
    f = pynuml.io.File("/sps/lbno/zappacosta/v10_11/nugraph/pynuml/tests/MyFile_r20000001_s00000_ts916850.h5")
    processor = pynuml.process.HitGraphProducer(
        file=f,
        semantic_labeller=pynuml.labels.StandardLabels(),
        event_labeller=pynuml.labels.FlavorLabels(),
        label_vertex=True)
    plot = pynuml.plot.GraphPlot(
        planes=["u", "v", "y"],
        classes=pynuml.labels.StandardLabels().labels[:-1])
    f.read_data_all(use_seq_cnt=False, evt_part=0)
    evts = f.build_evt()
    for evt in evts:
        _, data = processor(evt)
        if not data:
            continue
        plot.plot(data, target='semantic', how='true', filter='show')
        plot.plot(data, target='instance', how='true', filter='true')
    
    print("Sei sul cluster mona, che te plotti")
    
    #plot.SaveAs("/sps/lbno/zappacosta/v10_11/nugraph/pynuml/tests/tot.html")

#def main():
#    test_process_uboone()
#    print("Sei sul cluster mona, che te plotti")

#if __name__ == "__main__":
#    main()

# def test_process_dune_nutau():
#     """Test graph processing with DUNE beam nutau dataset"""
#     f = pynuml.io.File("/raid/nugraph/dune-nutau/test.evt.h5")
#     processor = pynuml.process.HitGraphProducer(
#         file=f,
#         semantic_labeller=pynuml.labels.StandardLabels(),
#         event_labeller=pynuml.labels.FlavorLabels(),
#         label_position=True)
#     plot = pynuml.plot.GraphPlot(
#         planes=["u", "v", "y"],
#         classes=pynuml.labels.StandardLabels().labels[:-1])
#     f.read_data(0, 100)
#     evts = f.build_evt()
#     for evt in evts:
#         _, data = processor(evt)
#         if not data:
#             continue
#         plot.plot(data, target="filter", how="true", filter="show")
#         plot.plot(data, target='semantic', how='true', filter='show')
#         plot.plot(data, target='instance', how='true', filter='true')
#         plot.plot(data, target="semantic", how="true", filter="show", xyz=True)
