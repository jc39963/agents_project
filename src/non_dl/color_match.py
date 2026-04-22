from colorharmonies import Color, triadicColor

def color_theory_test(rgb: list[float]) -> list[list]:
    """Uses colorharmonies library to find color triad given a specific RGB code.

    Args:
        rgb (list[float]): Input RGB code that we want to find the two triadic colors for.

    Returns:
        list[list]: List of the two triadic colors in RGB format.
    """
    rgb_conv = [int(x) for x in rgb]
    color = Color(rgb_conv, "", "")
    matches = triadicColor(color)
    return matches



# print(color_theory_test([0, 255, 255]))

# tools required: 
# #categorize image (load svm & pca, do preprocessing on saved image and get category)
# get color of item in image
# get matching colors
# get matching items
# filter catalog for matching items
# on those items, do cosine similarity between item
# and feature vecs, and between desired colors and catalog colors
