from .utils import *


def find_recs(
    rgb_list: list[tuple], item_list: list[str], feat_vec: list[float]
) -> list[str]:
    """Uses the feature vector for an item to find similar items in catalog. Then scores resulting
    items based on how well their colors match the desired triadic color scheme for the item.
    Calculates composite scores and returns the items with the 2 highest scores, for eaching of the desired
    item types.

    Args:
        rgb_list (list[tuple]): List of two RGB codes for the colors that make up a triadic combination with the color of the target item.
        item_list (list[str]): List of two item types that pair well with the target item.
        feat_vec (list[float]): Feature vector for the target item.

    Returns:
        list[str]: List of item IDs that are recommended based on the target item.
    """
    assert len(item_list) == len(rgb_list)
    features_index = get_db("product-non-dl-features")
    color_index = get_db("product-non-dl-colors")
    items_to_return = []
    for idx in range(len(item_list)):
        desired_item = item_list[idx]
        desired_rgb = rgb_list[idx]

        try:
            feat_match_results = features_index.query(
                vector=feat_vec, top_k=5, filter={"item_type": desired_item}
            )

            feat_scores = {x["id"]: x["score"] for x in feat_match_results["matches"]}
            print(f"Got items based on feature vector search...")
            if not feat_scores:
                continue

            ids = list(feat_scores.keys())
            color_search = color_index.fetch(ids=ids)

            color_scores = {}
            desired_array = np.array(desired_rgb)
            for id, vec in color_search.vectors.items():
                rgb = np.array(vec.values)
                color_score = euclidean_similarity_score(rgb, desired_array)
                color_scores[id] = color_score

            weighted_scores = {}
            for id in feat_scores:
                weighted = 0.7 * feat_scores[id] + 0.3 * color_scores.get(id, 0)
                weighted_scores[id] = weighted

            lowest_two_scores = sorted(
                weighted_scores, key=weighted_scores.get, reverse=True
            )[:2]
            items_to_return.extend(lowest_two_scores)
        except Exception as e:
            print(
                f"Feature vector query failed for {desired_item}. Error: {e}. Performing color-only search."
            )
            try:
                color_match_results = color_index.query(
                    vector=desired_rgb, top_k=2, filter={"item_type": desired_item}
                )
                items_to_return.extend(
                    [x["id"] for x in color_match_results["matches"]]
                )
            except Exception as fallback_e:
                print(f"Color fallback query failed: {fallback_e}")
    return items_to_return
