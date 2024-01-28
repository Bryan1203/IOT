import numpy as np
from collections import deque

def bfs_path_planning(map_array, start, goal):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    if map_array[start] == 1 or map_array[goal] == 1:
        return None 
    
    queue = deque([(start, [start])])
    
    visited = set()
    visited.add(start)
    
    while queue:
        current_position, path = queue.popleft()
        
        if current_position == goal:
            return path  
        
        for direction in directions:
            row_offset, col_offset = direction
            next_position = (current_position[0] + row_offset, current_position[1] + col_offset)
            
            if (0 <= next_position[0] < map_array.shape[0] and
                0 <= next_position[1] < map_array.shape[1] and
                map_array[next_position] == 0 and
                next_position not in visited):
                
                queue.append((next_position, path + [next_position]))
                visited.add(next_position)
    
    return None 