# For simplicity, we assume no dividends / rates / repo rates,
# as well as a flat volatility surface.

def validateVol(vol):
    # Linear Volatility
    if type(vol) is float:
        if vol <= 0.0:
            raise Exception("validateVol : Volatility cannot be negative")
    # Non-Linear Volatility = (vol_bid, vol_ask)
    elif type(vol) is tuple:
        if len(vol) != 2 or type(vol[0]) is not float or type(vol[1]) is not float:
            raise Exception("validateVol : Non-Linear Volatility must have exactly 2 float elements"
                            " (vol_bid, vol_ask)")
        if not (0.0 < vol[0] <= vol[1]):
            raise Exception("validateVol : Non-Linear Volatility must satisfy 0 < vol_bid <= vol_ask")
    else:
        raise Exception("validateVol : Unsupported type for Volatility")

def validateSpot(spot):
    if spot <= 0.0:
        raise Exception("validateSpot : Spot cannot be negative")

class Underlying:
    def __init__(self, spot, vol):
        self.is_non_linear = None
        self.setSpot(spot)
        self.setReferenceSpot(spot)
        self.setVol(vol)
        self.setReferenceVol(vol)
    # Setters
    def setSpot(self, spot):
        validateSpot(spot)
        self.spot = spot
    def setVol(self, vol):
        validateVol(vol)
        self.vol = vol
        self.is_non_linear = type(vol) is tuple
    def setReferenceVol(self, reference_vol):
        validateVol(reference_vol)
        self.reference_vol = reference_vol
        # The reference volatility is used solely to build the meshing: the mid volatility is enough.
        if type(reference_vol) is tuple:
            self.reference_vol = 0.5 * (reference_vol[0] + reference_vol[1])
    def setReferenceSpot(self, reference_spot):
        validateSpot(reference_spot)
        self.reference_spot = reference_spot
    # Getters
    def getSpot(self):
        return self.spot
    def getVol(self):
        return self.vol
    def getReferenceSpot(self):
        return self.reference_spot
    def getReferenceVol(self):
        return self.reference_vol
    def isNonLinear(self):
        return self.is_non_linear