# To create a new feature extractor, just copy this file and
# provide a sensible implementation for these two methods.
# Dump your new file inside the extractors folder and these
# methods will be called automatically for every training image
# when you run the feature manager program (fe_manager.py).

def getFeatureName():
	"""
	Returns the name of this feature. For ex., "Contrast" 
	or "Composition" or something. Must be unique.
	"""
	return "Test"

def extractFeature(img):
	""" 
	Returns the value of this feature computed for the 
	given image.
	"""
	return 0