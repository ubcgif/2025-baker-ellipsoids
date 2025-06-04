# definitions of ellipsoid classes to pass into the functions

class TriaxialEllipsoid:
    
    def __init__(self, a, b, c, yaw, pitch, roll, origin):
        
        # semiaxes 
        self.a = a # major_axis
        self.b = b # intermediate_axis
        self.c = c # minor_axis
        
        # euler angles
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        
        # origin
        self.origin = origin 
        
    

class ProlateEllipsoid:
    
    def __init__(self, a, b, yaw, pitch, origin):
        
        # semiaxes 
        self.a = a # major_axis
        self.b = b # minor axis
        
        
        # euler angles
        self.yaw = yaw
        self.pitch = pitch
        
        # origin
        self.origin = origin 
        
    @property 
    def c(self):
        return self.b
        

class OblateEllipsoid:
    
    def __init__(self, a, b, yaw, pitch, origin):
        
        # semiaxes 
        self.a = a # minor ais
        self.b = b # major axis
        
        
        # euler angles
        self.yaw = yaw
        self.pitch = pitch
        
        # origin
        self.origin = origin 
        
    @property 
    def c(self):
        return self.b