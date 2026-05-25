from .base import *

def are_the_labels_all_the_same_len(labels_list: list[np.ndarray]):
    length = labels_list[0].shape[0]
    for label in labels_list:
        if length != label.shape[0]:
            return False
    return True

def assign_cluster_id_to_array_in_place(row_index_to_cluster_id : np.ndarray, 
                                        labels_list : list[np.ndarray], 
                                        abstained_label_per_labels : list,
                                        first_compare_index : int, 
                                        start_index : int,
                                        end_index : int, 
                                        current_cluster_id : int,
                                        threshold : float):
    '''
    Takes the row_index_to_cluster_id and modifies it in-place, assigning the current_cluster_id to all the
    successive rows to the start_index until end_index, if and only if that row has the same label 
    of label[first_compare_index].

    The belonging to a cluster is decided by majority vote 
    '''
    labels_amount = float(len(labels_list)) # The amount of clustering algorithms used, inferred by how many labels are received
    
    for comparative_index in range(start_index, end_index):
        votes_for_same_cluster = 0.0 # How many clustering algorithms vote for the two rows to be in the same cluster
        # Find the vote of each clustering algorithm
        for i in range(len(labels_list)):
            label = labels_list[i]
            
            current_index_cluster_id_in_this_label = label[first_compare_index]
            comparative_index_cluster_id_in_this_label = label[comparative_index]

            # Are they assigned to the same cluster?
            if current_index_cluster_id_in_this_label == comparative_index_cluster_id_in_this_label:
                votes_for_same_cluster += 1 # Increase the votes in favor
             
        # Once votes are given, get the final verdict
        if votes_for_same_cluster/labels_amount >= threshold:
            # Same clusters, so, fill the final array
            row_index_to_cluster_id[comparative_index] = current_cluster_id


def compute_ensemble_labels(labels_list: list[np.ndarray],
                            threshold: float = 0.5) -> np.ndarray:
    '''

    Assign a cluster id to all of the rows of the dataframe, starting by labels_list which is
    a list of all the labels assigned by multiple clustering algorithms, by majority vote.

    The space complexity is O(n), n being the rows amount, while time complexity is O(clusters_num * n) = O(n).
    This is because the iteration over the rows is done once per cluster, as for each iteration an entire
    cluster is assigned to the components of it.

    If cluster_num is close to n, it becomes O(n^2) which is highly infeasible, but this scenario is very
    unreasonable to happen.

    You should always use threads_number = 1.

    params:
        labels_list: list of the labels assigned by each clustering algorithm. One label per clustering algorithm.
        abstained_label_per_labels: list of the label that, for each algorithm, states that a row has no cluster assigned to it.
                                    for example, hbscan would have -1, converted to the correct type used.
                                    If no abstained label is used for an algorithm, use None in that index.
                                    If no abstained labels are used at all, use None for this parameter entirely.
        threshold: used to assign a cluster to a row. each algorithm votes, and then the threshold is compared to the mean vote

    '''

    # Check if the labels all have the same length
    if not are_the_labels_all_the_same_len(labels_list=labels_list):
        raise ValueError(f"The labels have different lengths. Cannot ensemble: {[label.shape[0] for label in labels_list]}")

    rows_amount = labels_list[0].shape[0] # Number of rows in the dataframe.
    
    sentinel = np.iinfo(np.int32).min # The value that is used as a filler to mark a "unfilled" array cell
                                      # Cannot use -1 to fill it, because it is used by some algorithms.
    row_index_to_cluster_id = np.full(rows_amount, sentinel, dtype=np.int32) # Array that maps a row index to its cluster id

    next_clust_id = np.int32(0) # The id to assign to the next cluster, if a new one is found

    # Iterate over each row by index, and assign its cluster instantly
    for current_index in range(rows_amount):
        current_cluster_id = row_index_to_cluster_id[current_index] # Get the current cluster id, if it was assigned
        # If no cluster id is given, assign it the next cluster id, which initiates a new cluster
        if current_cluster_id == sentinel:
            current_cluster_id = next_clust_id # Assign the correct current cluster id for the iteration
            row_index_to_cluster_id[current_index] = current_cluster_id # Fill the final array in the current index p
            next_clust_id += 1 # Increase the cluster id for a future new cluster
        else:
            # A cluster id has been assigned already, I can skip this row
            continue

        # Assign the cluster of the next elements of row_index_to_cluster_id, if and only if
        # they are assigned to the same cluster by MAJORITY vote
        assign_cluster_id_to_array_in_place(
            row_index_to_cluster_id=row_index_to_cluster_id, 
            labels_list=labels_list,
            first_compare_index=current_index,
            start_index=current_index+1, 
            end_index=rows_amount, 
            current_cluster_id=current_cluster_id,
            threshold=threshold
        )
    
    return row_index_to_cluster_id

def make_ensemble_cluster_fn(
    cluster_fns: list[ClusterFn],
    threshold: float = 0.5,
) -> ClusterFn:
    """Compose multiple ClusterFns via consensus. Returns a single ClusterFn."""

    def _fn(X: np.ndarray) -> np.ndarray:
        labels_list = [cluster_fn(X) for cluster_fn in cluster_fns]
        
        labels = compute_ensemble_labels(labels_list=labels_list, threshold=threshold)
        return labels

    return _fn

