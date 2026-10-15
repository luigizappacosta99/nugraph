from .standard import StandardLabels

class DuneFDVDLabels(StandardLabels):
    '''Semantic labels for DUNE FD VD.

    Same particle classes as StandardLabels, but diffuse activity and
    background (hits with no associated true particle) are merged into a
    single "noise" class at index 6. "invisible" stays last, so that it is
    still dropped by labels[:-1] when building the semantic class metadata.

    Because the standard labelling function assigns every diffuse particle
    through self.diffuse, aliasing diffuse to the noise index is enough to
    route all of it into the noise class.
    '''

    def __init__(self,
                 gamma_threshold: float = 0.02,
                 hadron_threshold: float = 0.2):
        super(DuneFDVDLabels, self).__init__(gamma_threshold, hadron_threshold)

        self._labels = [
            'pion',
            'muon',
            'kaon',
            'hadron',
            'shower',
            'michel',
            'noise',
            'invisible'
        ]

    @property
    def noise(self):
        return self.index('noise')

    @property
    def diffuse(self):
        # diffuse activity is part of the noise class
        return self.noise

    @property
    def background(self):
        # hits with no true particle (semantic label -1 in the standard
        # scheme) are also part of the noise class
        return self.noise