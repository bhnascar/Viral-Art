import average_hsv
import avg_roughness
import color_palette_feature
import palette_contrast

functions = [
	average_hsv.extractFeature,
	avg_roughness.extractFeature,
	color_palette_feature.extractFeature,
	palette_contrast.extractFeature
]

names = [
	average_hsv.getFeatureName,
	avg_roughness.getFeatureName,
	color_palette_feature.getFeatureName,
	palette_contrast.getFeatureName
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
