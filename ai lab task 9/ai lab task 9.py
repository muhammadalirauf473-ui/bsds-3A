import math

def minimax(depth, index, isMaxPlayer, values, maxDepth):
    if depth == maxDepth:
        return values[index]

    if isMaxPlayer:
        left = minimax(depth + 1, index * 2, False, values, maxDepth)
        right = minimax(depth + 1, index * 2 + 1, False, values, maxDepth)
        return max(left, right)
    else:
        left = minimax(depth + 1, index * 2, True, values, maxDepth)
        right = minimax(depth + 1, index * 2 + 1, True, values, maxDepth)
        return min(left, right)

leaf_values = [3, 5, 2, 9]
depth_of_tree = int(math.log2(len(leaf_values)))

result = minimax(0, 0, True, leaf_values, depth_of_tree)
print("Optimal value is:", result)