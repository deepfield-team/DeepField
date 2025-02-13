"""Table meta-information required for parsing"""

_ATM_TO_PSI = 14.6959

TABLE_INFO = {
    'PVTO': dict(attrs=['RS', 'PRESSURE', 'FVF', 'VISC'], domain=[0, 1],
                 defaults=None),

    'PVTG': dict(attrs=['PRESSURE', 'RV', 'FVF', 'VISC'], domain=[0, 1],
                 defaults=None),

    'PVDG': dict(attrs=['PRESSURE', 'FVF', 'VISC'], domain=[0],
                 defaults=None),

    'PVDO': dict(attrs=['PRESSURE', 'FVF', 'VISC'], domain=[0],
                 defaults=None),

    'PVTW': dict(attrs=['PRESSURE', 'FVF', 'COMPR', 'VISC', 'VISCOSIBILITY'], domain=[0],
                 defaults=[(1, _ATM_TO_PSI), 1, (4e-5, 4e-5/_ATM_TO_PSI), 0.3, 0]),

    'PVCDO': dict(attrs=['PRESSURE', 'FVF', 'COMPR', 'VISC', 'VISCOSIBILITY'], domain=[0],
                 defaults=[None, None, None, None, 0]),

    'SWOF': dict(attrs=['SW', 'KRWO', 'KROW', 'POW'], domain=[0],
                 defaults=[None, None, None, 0]),

    'SGOF': dict(attrs=['SG', 'KRGO', 'KROG', 'POG'], domain=[0],
                 defaults=[None, None, None, 0]),

    'RSVD': dict(attrs=['DEPTH', 'RS'], domain=[0],
                 defaults=None),

    'ROCK': dict(attrs=['PRESSURE', 'COMPR'], domain=[0],
                 defaults=[(1.0132, 1.0132*_ATM_TO_PSI), (4.934e-5, 4.934e-5/_ATM_TO_PSI)]),

    'DENSITY': dict(attrs=['DENSO', 'DENSW', 'DENSG'], domain=None,
                    defaults=[(600, 37.457),  (999.014, 62.366), (1, 0.062428)])

}
