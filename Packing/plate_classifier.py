import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
class Plate_Classifier:
    def __init__(self,path="plate_polygon.csv"):
        df = pd.read_csv(path)
        latitude = df['X']
        longitude = df['Y']
        classes = df['Plate']
        unique_classes = np.unique(classes)
        polygon_ls = list()
        polygon_ls_org = list()
        for cls in unique_classes:
            polygon = list()
            polygon_org = list()
            for i in range(len(classes)):
                if(classes[i]==cls):
                    polygon.append(Point(latitude[i],longitude[i]))
                    polygon_org.append([latitude[i],longitude[i]])
            polygon_ls.append(polygon)
            polygon_ls_org.append(polygon_org)
        self.polygon_ls = polygon_ls
        self.polygon_ls_org = polygon_ls_org
        self.num_plate = len(unique_classes)


    #You can get a one-hot plate assignment by giving latitude and longitude as a input
    def plate(self,lon,lat):
        result = [0 for i in range(self.num_plate)]
        for i in range(self.num_plate):
            result[i]=self.point_in_polygon(Point(lon,lat),self.polygon_ls[i])
        result1 = [False for i in range(self.num_plate-1)]
        if(result[0]==True or result[1]==True):
            result1[0] = True
        if(result[11]==True or result[12]==True):
            result1[11]=True
        for i in range(1,self.num_plate-1):
            result1[i] = result[i+1]
        return result1


    #You may visualize the plate by csv file. Check the idx for the name of tetical plate.
    def visualize(self,point_csv = "all.csv",idx=0):
        df = pd.read_csv(point_csv)

        # Extract latitude and longitude columns
        latitude = df['lat']
        longitude = df['lon']
        classes = df['lon']
        unique_classes = np.unique(classes)


        num_unique_classes = len(unique_classes)
        colors = plt.cm.tab10(np.linspace(0, 1, num_unique_classes))  # Using a colormap to generate distinct colors
        fig,ax = plt.subplots()
        # Plot the points with different colors for each class
        for class_label, color in zip(unique_classes, colors):
            class_indices = classes == class_label
            ax.scatter(longitude[class_indices], latitude[class_indices], color=color, label=class_label, alpha=0.5)
        
       

        for i in range(len(self.polygon_ls_org)):
            y = np.array(self.polygon_ls_org[i])
            p = Polygon(y,facecolor='k')
            ax.add_patch(p)

        
        # Plot the points
        # plt.scatter(longitude, latitude, color='blue', alpha=0.5)  # alpha adjusts the transparency
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Figure of Points')
        plt.grid(True)
        plt.show()

    # Checking if a point is inside a polygon
    def point_in_polygon(self,point, polygon):
        num_vertices = len(polygon)
        x, y = point.x, point.y
        inside = False
    
        # Store the first point in the polygon and initialize the second point
        p1 = polygon[0]
    
        # Loop through each edge in the polygon
        for i in range(1, num_vertices + 1):
            # Get the next point in the polygon
            p2 = polygon[i % num_vertices]
    
            # Check if the point is above the minimum y coordinate of the edge
            if y > min(p1.y, p2.y):
                # Check if the point is below the maximum y coordinate of the edge
                if y <= max(p1.y, p2.y):
                    # Check if the point is to the left of the maximum x coordinate of the edge
                    if x <= max(p1.x, p2.x):
                        # Calculate the x-intersection of the line connecting the point to the edge
                        x_intersection = (y - p1.y) * (p2.x - p1.x) / (p2.y - p1.y) + p1.x
    
                        # Check if the point is on the same line as the edge or to the left of the x-intersection
                        if p1.x == p2.x or x <= x_intersection:
                            # Flip the inside flag
                            inside = not inside
    
            # Store the current point as the first point for the next iteration
            p1 = p2
    
        # Return the value of the inside flag
        return inside
 
# Driver code
if __name__ == "__main__":
    # Define a point to test
    classify = Plate_Classifier()
    print(classify.plate(100,50))
    classify.visualize()
    


# 0	PacificWest
# 1	PacificEast
# 2	Phillippine
# 3	North American
# 4	Eurasian
# 5	African
# 6	Arabian
# 7	Indian
# 8	Somali
# 9	Southe American
# 10	Nazca
# 11	Antartic
# 12 AustralianWest
# 13 AustralianEast