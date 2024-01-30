from picar_4wd.path_planning import bfs_path_planning
from picar_4wd.motor import Motor


def navigate(obstacle_map, start, goal):
    path = bfs_path_planning(obstacle_map, start, goal)
    if path:
        # Create a Motor instance and follow the path
        motor = Motor()
        follow_path(path, motor)
    else:
        print("No path found")
