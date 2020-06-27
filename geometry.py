import numpy as np

def is_inside(point, vertices):
    """
    Test if a point is in a polygon
    """
    number_vertices = len(vertices)
    j =  -1
    oddNodes = False
    for i in range(number_vertices):
        if((vertices[i,1] < point[1] and vertices[j,1] >= point[1]) or (vertices[j,1] < point[1] and vertices[i,1] >= point[1])):
            # find intersection check left or right
            if(vertices[i,0] + (point[1] - vertices[i,1])/(vertices[j,1] - vertices[i,1])*(vertices[j,0] - vertices[i,0]) < point[0]): 
                oddNodes = not oddNodes
        j = i
    return oddNodes


def tri_orientation(a, b, c):
    precision = 10**(-6)
    #cross_prodcut
    overlap_vector = (c[1] - a[1])*(b[0] - a[0]) - (b[1]-a[1])*(c[0]-a[0])
    if(abs(overlap_vector) < precision): return 0
    elif (overlap_vector > 0): return 1
    else: return 2


def segment_intersect(a, b, c, d):
    o1 = tri_orientation(a,c,d)
    o2 = tri_orientation(b,c,d)
    o3 = tri_orientation(a,b,c)
    o4 = tri_orientation(a,b,d)
    
    if(o1!=o2 and o3!=o4):
        return True
    
    if(o1==o2==o3==o4==0):
        # all points in a line
        vector1 = b - a
        p1 = np.dot(a, vector1)
        p2 = np.dot(b, vector1)
        min_1 = min(p1,p2)
        max_1 = max(p1,p2)
        
        p3 = np.dot(c, vector1)
        p4 = np.dot(d, vector1)
        
        if((p3 > min_1 and p3 < max_1) or (p4 > min_1 and p4 < max_1)):
            return True
    
    return False

def rotate_2d_vector(vector, angle):
    c, s = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[c,-s],[s,c]])
    return rot_matrix@vector


def test_simple_polygon_overlap(polygon1, polygon2, boxSize):
    dr = polygon2.position - polygon1.position
    for i in range(2):
        if dr[i] < -0.5*boxSize:
            dr[i] += boxSize
        elif dr[i] >= 0.5*boxSize:
            dr[i] -= boxSize
    #vert1 = polygon1.vertices
    #vert2 = polygon2.vertices
    
    #points in 1 rotate to normal coordinate and then rotate into 2's coordinates
    ab_rotation = -polygon2.orientation%(np.pi*2) + polygon1.orientation%(np.pi*2)
    # rotate dr in normal coordinate into 2's coordinate system
    ab_t = rotate_2d_vector(dr, -polygon2.orientation)
    
    j = -1
    prev_a = rotate_2d_vector(polygon1.vertices[-1], ab_rotation) - ab_t
    for i in range(polygon1.N):
        cur_a = rotate_2d_vector(polygon1.vertices[i], ab_rotation) - ab_t
        
        k = -1 
        prev_b = polygon2.vertices[k]
        for l in range(polygon2.N):
            cur_b = polygon2.vertices[l]
            if(segment_intersect(prev_a, cur_a, prev_b, cur_b)):
                return True
            k = l;
            prev_b = cur_b
        
        if(is_inside(cur_a, polygon2.vertices)):
            return True
        
        j = i
        prev_a = cur_a
        
    
    ab_rotation = -ab_rotation
    ab_t = rotate_2d_vector(-dr, -polygon1.orientation)
    
    for i in range(polygon2.N):
        cur = rotate_2d_vector(polygon2.vertices[i], ab_rotation) - ab_t
        
        if(is_inside(cur, polygon1.vertices)):
            return True
    
    return False
    

class Square:
    def __init__(self, position, orientation, side_length = 1,  pt_type = 'diag'):
        if pt_type == 'diag':
            self.vertices = side_length * np.array([[0,np.sqrt(0.5)],[np.sqrt(0.5),0],[0,-np.sqrt(0.5)],[-np.sqrt(0.5), 0 ]])
        elif pt_type == 'edge':
            self.vertices = side_length * np.array([[-0.5,0.5],[0.5,0.5],[0.5,-0.5],[-0.5,-0.5]])
        self.position_double = position
        self.position = np.array([position[0], position[1]])
        self.orientation = orientation
        self.N = len(self.vertices)
        self.diameter = side_length*np.sqrt(2)
        
    def get_vertices(self):
        res = list()
        for vert in self.vertices:
            res.append(rotate_2d_vector(vert, self.orientation) + self.position)
        return np.array(res)
    
    def get_raduis(self):
        return np.hypot(self.position[0], self.position[1])