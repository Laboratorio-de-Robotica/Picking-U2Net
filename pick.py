from u2net_predict import U2netModel
import argparse
import numpy as np
import cv2 as cv
from types import SimpleNamespace

class PickU2Net:
  '''
  PickU2Net class
  This class is used to choose grabbing points for objects detected in an image using a U2Net model.
  Constructor let choose the model to use (u2net or u2netp).
  __call__ method is used to analyze the input image and get the grabbing points.
  annotate method is used to annotate the input image with the grabbing points.
  analizeContour method is used internally on one contour to get the gravity center and principal component unity vector of a contour.
  getGrabPoints method is used internally on one contour to get the two grabbing points.

  Properties:
    model: U2netModel object
    minArea: int, minimum area of a contour to be considered
    maxArea: int, maximum area of a contour to be considered
    map: np.array, binary map of the output image
    contours: list of np.array, list of contours detected in the map
    hierarchy: np.array, hierarchy of the contours
    results: list of results of the analysis with these properties:
      center: tuple of int, gravity center of the contour
      principalComponent: np.array, principal component unity vector of the contour
      grabbingPoint0: tuple of int, first grabbing point
      grabbingPoint1: tuple of int, second grabbing point
      contour: np.array, contour analyzed

    A results element can be None if the associated contour is rejected.
  '''
  def __init__(self, model_name="u2net", minArea=100, maxArea=100000):
    self.model = U2netModel(model_name)
    self.minArea = minArea
    self.maxArea = maxArea

  def __call__(self, input_image):
    output_image = self.model(input_image)
    self.map = cv.inRange(output_image, 128, 255)  # You can adjust threshold (inRange 2nd argument)
    self.contours, self.hierarchy = cv.findContours(self.map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # cv.RETR_CCOMP for 2 level (contours with their holes)
    self.results = []

    # Analize and annotate contours
    for contour in self.contours:
      # Filter contours
      area = cv.contourArea(contour)
      if(area<self.minArea or area>self.maxArea):
        # Too small or too big
        self.results.append(None)
        continue

      # gravity center c, principal component unity vector v
      center, principalComponent = self.analizeContour(contour)
      grabbingPoint0, grabbingPoint1 = self.getGrabPoints(contour, center, principalComponent)
      result = SimpleNamespace()
      result.center = center
      result.principalComponent = principalComponent
      result.grabbingPoint0 = grabbingPoint0
      result.grabbingPoint1 = grabbingPoint1
      result.contour = contour
      self.results.append(result)

    return self.results
  

  def annotate(self, input_image):
    # Gray image for color annotation
    imVis = cv.cvtColor(cv.cvtColor(input_image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR)

    # Analize and annotate contours
    for result in self.results:
      if(not result):
        continue

      # Annotation
      cv.drawContours(imVis, [result.contour], 0, (0, 128, 255), 2)
      arrow = (result.principalComponent*50).astype(np.int32)
      cv.arrowedLine(imVis, result.center, result.center + arrow, (255, 128, 0), 3, tipLength=0.2)

      cv.circle(imVis, result.grabbingPoint0, 5, (0,0,255), 3)
      cv.circle(imVis, result.grabbingPoint1, 5, (0,0,255), 3)

    return imVis

  # Gets gravity center and principal component unity vector
  def analizeContour(self, contour):
    moments = cv.moments(contour)
    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    M = np.array([[moments['mu20'], moments['mu11']],[moments['mu11'], moments['mu02']]])
    eigenvalues, eigenvectors = np.linalg.eig(M)  # Always two eigen
    if(eigenvalues[0]>eigenvalues[1]):
      principalComponent = eigenvectors[:, 0]
    else:
      principalComponent = eigenvectors[:, 1]
    return center, principalComponent

  # Determines two grabing points p1 and p2
  # Countour intersection with v's perpendicular on c
  # It doesn't check if more than two point are intersected,
  # nor check the slipery when contact points aren't normal to the gripper finger
  def getGrabPoints(self, contour, center, principalComponent):
    contour = np.squeeze(contour)
    distancesFromEdge = np.dot(contour - center, principalComponent)
    zeroDistanceIndices = np.argwhere(np.abs(distancesFromEdge)<1.0)
    middleIndex = int((zeroDistanceIndices[0]+zeroDistanceIndices[-1])/2)

    p0Indices = zeroDistanceIndices[zeroDistanceIndices < middleIndex]
    p0Points = contour[p0Indices]
    p0 = np.average(p0Points, axis=0).astype(np.int32)

    p1Indices = zeroDistanceIndices[zeroDistanceIndices > middleIndex]
    p1Points = contour[p1Indices]
    p1 = np.average(p1Points, axis=0).astype(np.int32)
    
    return p0,p1


# Main
if __name__ == "__main__":
  # Parsing arguments from command line
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", help="input image file path", default="images/imagen_13r.jpeg")
  parser.add_argument("-o", "--output", help="output image file path", default="")
  parser.add_argument("-m", "--model", help="model, either u2net (default) or u2netp", default="u2net")
  args = parser.parse_args()

  model_name = args.model
  input_image_path = args.input
  output_image_path = args.output

  # Read image and analyze
  input_image = cv.imread(input_image_path)
  picking = PickU2Net(model_name)
  results = picking(input_image)
  imVis = picking.annotate(input_image)

  # Show results
  cv.imshow('Map', picking.map)
  cv.imshow('Grasp', imVis)
  cv.waitKey(0)
