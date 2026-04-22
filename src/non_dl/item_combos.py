def create_combo(item_type: str) -> list[str]:
    """Returns two item types that pair well with a particular input item of clothing.

    Args:
        item_type (str): Item we want to find matching clothes for.

    Returns:
        list[str]: Items that pair well with the input item.
    """
    item_groups = {
       "sunglass": ["tops", "jeans"],
       "hat": ["jackets", "jeans"],
       "jacket": ["tops", "pants"],
       "shirt": ["pants", "jackets"],
       "pants": ["tops", "jackets"],
       "shorts": ["tops", "jackets"],
       "skirt": ["jackets", "tops"],
       "dress": ["jackets", "tops"],
       "bag": ["skirts", "dresses"],
       "shoe": ["dresses", "pants"]
    }
    return item_groups[item_type]