
class TriaxialEllipsoid:
    
    def __init__(self, semiaxes, euler_angles, origin):
        
        # semiaxes 
        self.a = semiaxes[0] # major_axis
        self.b = semiaxes[1] # intermediate_axis
        self.c = semiaxes[2] # minor_axis
        
        # euler angles
        self.yaw = euler_angles[0]
        self.pitch = euler_angles[1]
        self.roll = euler_angles[2]
        
        # origin
        self.origin = origin 
        
    
ellipsoid1 = Triaxial((5,4,3), (0, 30, 0), (0,0))

print(ellipsoid1.a)
class ProlateEllipsoid:
    
    def __init__(self, a, b, c, yaw, pitch, origin):
        
        self.a = a # major_axis
        self.b = b # minor_axis (b=c)
        
        # euler angles
        self.yaw = euler_angles[0]
        self.pitch = euler_angles[1]
        self.roll = 0 # roll has no effect as b=c
        
        # origin
        self.origin = origin 
        
        @property 
        def c(self):
            return self.b
        

class OblateEllipsoid:
    
    def __init__(self, semiaxes, euler_angles, origin):
        
        # semiaxes 
        self.a = semiaxes[0] # minor_axis
        self.b = semiaxes[1] # major_axis(b=c)
        self.c = semiaxes[1] # major_axis (b=c)
        
        # euler angles
        self.yaw = euler_angles[0]
        self.pitch = euler_angles[1]
        self.roll = 0 # roll has no effect as b=c
        
        # origin
        self.origin = origin 