import average_hsvl
import avg_roughness
import color_palette_feature
import face
import mode_hsvl
import palette_contrast
import segmentation
import torso

functions = [
	average_hsvl.extractFeature,
	avg_roughness.extractFeature,
	color_palette_feature.extractFeature,
	face.extractFeature,
	mode_hsvl.extractFeature,
	palette_contrast.extractFeature,
	segmentation.extractFeature,
	torso.extractFeature
]

names = [
	average_hsvl.getFeatureName,
	avg_roughness.getFeatureName,
	color_palette_feature.getFeatureName,
	face.getFeatureName,
	mode_hsvl.getFeatureName,
	palette_contrast.getFeatureName,
	segmentation.getFeatureName,
	torso.getFeatureName
]

# The clever way of doing this is not working yet... #

# from os.path import dirname, basename, isfile
# import glob

# # Find all python files in this directory.
# modules = glob.glob(dirname(__file__)+"/*.py")
# filenames = [basename(f)[:-3] for f in modules if isfile(f)]

# # Collects all the feature extraction functions from files
# # in this directory.
# functions = []

# # Collects all the feature names corresponding to each of
# # the feature extraction functions above.
# names = []

# for filename in filenames:
# 	if filename == "__init__":
# 		continue
# 	module = __import__(dirname(__file__) + "." + filename)
# 	submodule = getattr(module, filename)
# 	functions.append(getattr(submodule, "extractFeature"))
# 	names.append(getattr(submodule, "getFeatureName")())
