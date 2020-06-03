#coding: utf-8
import numpy as np
from PIL import Image

def showarr(bool_arr):
    pil = Image.fromarray((bool_arr * 255).astype(np.uint8))
    w, h = pil.size
    pil.save("map.png")

def is_valid_link(cross, y_link, x_link, n_road):
    # 行き止まりはだめ
    link_nums = np.logical_and(np.roll(cross, -1, axis = 0), np.roll(y_link, 0, axis = 0)).astype(np.int32) + \
                np.logical_and(np.roll(cross,  1, axis = 0), np.roll(y_link, 1, axis = 0)).astype(np.int32) + \
                np.logical_and(np.roll(cross, -1, axis = 1), np.roll(x_link, 0, axis = 1)).astype(np.int32) + \
                np.logical_and(np.roll(cross,  1, axis = 1), np.roll(x_link, 1, axis = 1)).astype(np.int32)
    #print((link_nums >= 2)[cross].all())
    #print(cross.astype(np.int32))
    #print(y_link.astype(np.int32))
    #print(x_link.astype(np.int32))
    #print(link_nums)
    if (link_nums >= 2)[cross].all():
        injc = get_shearing_2points(cross, n_road)
        if None != injc:
            if all_connected(cross, y_link, x_link, n_road, injc[0]):
               return True
    return False

def get_shearing_2points(cross, n_road):
    flat_pos_vec = np.where(cross.flatten())[0]

    points = []
    if flat_pos_vec.sum() >= 2:
        for flat_pos in flat_pos_vec:
            y = flat_pos // n_road
            x = flat_pos %  n_road
            if len(points) == 0:
                points.append((y, x))
            else:
                y_old, x_old = points[0]
                if (y != y_old) and (x != x_old):
                    points.append((y, x))
                    return points
    return None

def all_connected(cross, y_link, x_link, n_road, origin):
    reach = np.zeros((n_road, n_road), dtype = np.bool)
    origin_y, origin_x = origin
    reach[origin_y, origin_x] = True
    while True:
        update_any = False
        for propagate_dir, roll in zip([0, 1], [-1, 1]):
            for axis, link in zip([0, 1], [y_link,x_link]):
                exists_and_injected = np.logical_and(cross,reach)
                has_link = np.roll(link, propagate_dir , axis = axis)
                has_next = np.roll(cross, roll, axis = axis)
                next_not_injected = np.logical_not(np.roll(reach, roll, axis = axis))
                incr = np.logical_and(np.logical_and(exists_and_injected, has_link),
                                      np.logical_and(has_next, next_not_injected))
                if incr.any():
                    incr = np.roll(incr, -roll, axis = axis)
                    reach[incr] = True
                    update_any = True
        if update_any == False:
            break
    
    ok = reach[cross].all()
    return ok
        
def make_map_info(n_road,
                  cross_density,
                  road_density):
    map_trial = 0
    cross = np.random.uniform(size = (n_road, n_road)) <= cross_density
    while True:
        map_trial += 1
        y_link = (np.random.uniform(size = (n_road, n_road)) <= road_density)
        x_link = (np.random.uniform(size = (n_road, n_road)) <= road_density)
        if is_valid_link(cross, y_link, x_link, n_road):
            break
        print(map_trial)
    return cross, y_link, x_link

def main():
    n_road = 10
    cross_density = 0.8
    road_density = 1.0

    road_point = 10
    wall_point = 20  # even number
    cross, y_link, x_link = make_map_info(n_road, cross_density, road_density)
    map_size = (road_point + wall_point) * n_road
    map_flg = np.zeros((map_size, map_size), dtype = np.bool)
    for y in range(n_road):
        for x in range(n_road):
            if cross[y, x]:
                for yp in range(road_point):
                    for xp in range(road_point):
                        map_flg[(road_point + wall_point) * y + yp,
                                (road_point + wall_point) * x + xp] = True
                if y_link[y, x] and cross[(y + 1) % n_road, x]:
                    for yp in range(wall_point):
                        for xp in range(road_point):
                            map_flg[(road_point + wall_point) * y + road_point + yp,
                                    (road_point + wall_point) * x + xp] = True
                if x_link[y, x] and cross[y, (x + 1) % n_road]:
                    for yp in range(road_point):
                        for xp in range(wall_point):
                            map_flg[(road_point + wall_point) * y + yp,
                                    (road_point + wall_point) * x + road_point + xp] = True
    map_flg = np.roll(map_flg, (wall_point//2, wall_point//2), axis = (0,1))
    showarr(map_flg)

if __name__ == "__main__":
    main()
    print("Done")