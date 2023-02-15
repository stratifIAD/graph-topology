'''
Stratifiad project
@author : valentinabadie
@updates-spie : gabriel.jimenez
'''
import numpy as np
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import nearest_points
import pandas as pd

class MorphologyStudy():
    
    """This class computes different morphological variables for selected object (tangle or plaque) and can output them to a csv table.
    
    Attributes :
        labels : list of 2D lists, contours delimiting the retrieved objects
        gray_matter : list, gray matter contours
        metrics : dict, containing the different variables computed
    
    Methods :
    
        Built-in :
            __init__ : initializes the class
        
        Special :
            surface : computes the surface of all the objects of the slide
            perimeter : computes the perimeter of all the objects of the slide
            density : computes the density of objects of the slide (N objects/ square meter)
            circularity : computes the circularity of all the objects of the slide
            convex_hull_surface : computes the convex hull surface of all the objects of the slide
            convex_hull_perimeter : computes the convex hull perimeter of all the objects of the slide
            convexity : computes the convexity of all the objects of the slide
            roughness : computes the roughness of all the objects of the slide
            load : computes the density of objects of the slide (surface objects/ square meter)
            form_factor : computes the form factor of all the objects of the slide
            proximity : computes the proximity of all the objects of the slide
            output_all_variables : saves the variables in a csv table

    """
    
    def __init__(self, labels, slide_id, object_group = 3):
        
        """initializes the class
        
        Inputs :
            labels : list of 2D lists, contours delimiting the retrieved objects
            slide_id : ID of the slide on which segmentation has been performed
            object_group : 2 for tangles, 3 for plaques
        
        """
        self.slide_id = slide_id
        self.labels = [label for obj, label in list(filter(lambda x: x[0] != 1, labels))]
        self.object_group = object_group
        
        self.gray_matter = [label for obj, label in list(filter(lambda x: x[0] == 1, labels))]
        self.metrics = dict()
        
    def surface(self):
        """computes the surface of all the objects of the slide"""
        
        surfaces = list()
        for label in self.labels :
            surfaces.append(Polygon(list(zip(*label.T))).area)
        
        self.metrics['Surface'] = surfaces
        
    def perimeter(self):
        """computes the perimeter of all the objects of the slide"""
        
        perimeters = list()
        for label in self.labels :
            perimeters.append(Polygon(list(zip(*label.T))).length)
        
        self.metrics['Perimeter'] = perimeters
        
    def density(self):
        """computes the density of objects of the slide (N objects/ square meter)"""
        
        surfaces_gray_matter = list()
        for label in self.gray_matter :
            surfaces_gray_matter.append(Polygon(list(zip(*label.T))).area)
        
        self.metrics['Density'] = [len(self.labels)/sum(surfaces_gray_matter)]*len(self.labels)
    
    def circularity(self):
        """computes the circularity of all the objects of the slide"""
        
        if 'Surface' in self.metrics:
            surfaces = self.metrics['Surface']
        else :
            self.surface()
            surfaces = self.metrics['Surface']
            
        if 'Perimeter' in self.metrics:
            perimeters = self.metrics['Perimeter']
        else :
            self.perimeter()
            perimeters = self.metrics['Perimeter']
        
        self.metrics['Circularity'] = [4*np.pi*x/y**2 for x,y in list(zip(surfaces, perimeters))]
        
    def convex_hull_surface(self):
        """computes the convex hull surface of all the objects of the slide"""
        
        surfaces = list()
        for label in self.labels :
            surfaces.append(Polygon(list(zip(*label.T))).convex_hull.area)
        
        self.metrics['Convex hull surface'] = surfaces
        
    def convex_hull_perimeter(self):
        """computes the convex hull perimeter of all the objects of the slide"""
        
        perimeters = list()
        for label in self.labels :
            perimeters.append(Polygon(list(zip(*label.T))).convex_hull.length)
        
        self.metrics['Convex hull perimeter'] = perimeters
        
    def convexity(self):
        """computes the convexity of all the objects of the slide"""
        
        if 'Surface' in self.metrics:
            surfaces = self.metrics['Surface']
        else :
            self.surface()
            surfaces = self.metrics['Surface']
            
        if 'Convex hull surface' in self.metrics:
            convex_hull_surfaces = self.metrics['Convex hull surface']
        else :
            self.convex_hull_surface()
            convex_hull_surfaces = self.metrics['Convex hull surface']
        
        self.metrics['Convexity'] = [x/y for x,y in list(zip(surfaces, convex_hull_surfaces))]
        
    def roughness(self):
        """computes the roughness of all the objects of the slide"""
        
        if 'Perimeter' in self.metrics:
            perimeters = self.metrics['Perimeter']
        else :
            self.perimeter()
            perimeters = self.metrics['Perimeter']
            
        if 'Convex hull perimeter' in self.metrics:
            convex_hull_perimeters = self.metrics['Convex hull perimeter']
        else :
            self.convex_hull_perimeter()
            convex_hull_perimeters = self.metrics['Convex hull perimeter']
                  
        self.metrics['Roughness'] = [x/y for x,y in list(zip(perimeters, convex_hull_perimeters))]
        
    def load(self):
        """computes the density of objects of the slide (surface objects/ square meter)"""
        
        if 'Surface' in self.metrics:
            surfaces = self.metrics['Surface']
        else :
            self.surface()
            surfaces = self.metrics['Surface']
            
        if 'Density' in self.metrics:
            density = self.metrics['Density']
        else :
            self.density()
            density = self.metrics['Density']
        
        #print(np.mean(surfaces))
        #print(density)
        
        self.metrics['Load'] = np.mean(surfaces)*np.array(density)
        
    def form_factor(self):
        """computes the form factor of all the objects of the slide"""
        
        form_factors = list()
        for label in self.labels :
            poly = Polygon(list(zip(*label.T)))
            centroid = np.array(poly.centroid.coords)
            form_factors.append(np.sqrt(np.sum((centroid - label)**2)))
            
        self.metrics['Form factor'] = form_factors

    def proximity(self):
        """computes the proximity of all the objects of the slide"""

        distances = list()
        centroids = [list(Polygon(list(zip(*label.T))).centroid.coords)[0] for label in self.labels]       
                                    
        for i, centroid in enumerate(centroids):
            other_centroids = MultiPoint(centroids[:i]+centroids[i+1:])
            nearest_neighbor = list(nearest_points(other_centroids, Point(centroid))[0].coords)[0]
            distances.append(np.sqrt(np.sum((np.array(nearest_neighbor) - np.array(centroid))**2)))
                                    
        self.metrics['Proximity'] = distances

    def find_centroids(self):
        '''computes the centroids of each annotation'''

        centroids_coords_x = list()
        centroids_coords_y = list()
        for label in self.labels :
            coords = list(Polygon(list(zip(*label.T))).centroid.coords)[0]
            centroids_coords_x.append(coords[0])
            centroids_coords_y.append(coords[1])

        self.metrics['centroid-0'] = centroids_coords_x
        self.metrics['centroid-1'] = centroids_coords_y

    def output_all_variables(self, path_to_save=None):
        
        """Computes all the morphological variables and creates a frame out of them, and save it in a table
        
        Inputs :
            path_to_save : path to the file where the output table is to be saved
        """   
        
        #reinitialises the metrics dict
        self.metrics = dict()
        
        #adds slide, objects information
        self.metrics['Slide ID'] = [self.slide_id] * len(self.labels)
        self.metrics['Object ID'] = list(np.arange(1,len(self.labels)+1))
        self.metrics['Object Group'] = ['Tangles' if self.object_group == 2 else 'Neuritic plaques'] * len(self.labels)
        
        #computes all morphological variables relevant to the object group selected
        
        if self.object_group == 2:
            
            #comparison with a circle
            self.circularity()
            self.form_factor()
            
            #comparison with a convex object
            self.convexity()
            self.roughness()
            
            #glabal variable
            self.density()
            self.load()
            
            #semi-global variables
            self.proximity()

            #save centroids
            self.find_centroids()
            
        if self.object_group == 3 :
        
            #variables of size
            self.surface()
            self.perimeter()
            self.convex_hull_surface()
            self.convex_hull_perimeter()
            
            #comparison with a circle
            self.circularity()
            self.form_factor()
            
            #comparison with a convex object
            self.convexity()
            self.roughness()
            
            #glabal variable
            self.density()
            self.load()
            
            #semi-global variables
            self.proximity()

            #save centroids
            self.find_centroids()
        
        #self.heat_map()
        
        #creates the sheet
        df_morphology = pd.DataFrame.from_dict(self.metrics)
        
        #saves it in a csv file
        df_morphology.to_csv(path_to_save)

        return df_morphology, self.gray_matter
  
