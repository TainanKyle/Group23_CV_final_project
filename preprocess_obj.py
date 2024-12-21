
# input_file = 'data/scenes/room_tainan_1/meshes/scene.obj'
input_file = './meshed_simplified.obj'
output_file = './scene.obj'

# # flip the obj file along y axis
# with open(input_file, 'r') as file:
#     lines = file.readlines()

# new_lines = []
# for line in lines:
#     if line.startswith('v '):
#         parts = line.split()
#         x, y, z = map(float, parts[1:])
#         new_line = f'v {x} {-y} {z}\n'
#         new_lines.append(new_line)
#     else:
#         new_lines.append(line)

# with open(output_file, 'w') as file:
#     file.writelines(new_lines)



# # calculate the center of the obj file scene
# # read the obj file
# import numpy as np
# with open(input_file, 'r') as file:
#     lines = file.readlines()

# vertices = []
# for line in lines:
#     if line.startswith('v '):
#         parts = line.split()
#         x, y, z = map(float, parts[1:])
#         vertices.append([x, y, z])

# vertices = np.array(vertices)
# center = np.mean(vertices, axis=0)
# print(center)

# # calculate the maximum and minimum of y axis
# y_max = np.max(vertices[:, 1])
# y_min = np.min(vertices[:, 1])
# print(y_max, y_min)

# # calculate the maximum and minimum of x and z axis
# x_max = np.max(vertices[:, 0])
# x_min = np.min(vertices[:, 0])
# z_max = np.max(vertices[:, 2])
# z_min = np.min(vertices[:, 2])
# print(x_max, x_min, z_max, z_min)


# import open3d as o3d
# mesh = o3d.io.read_triangle_mesh(input_file)
# # print the number of faces
# # print(len(mesh.triangles))
# simplified_mesh = mesh.simplify_quadric_decimation(100000)
# o3d.io.write_triangle_mesh(output_file, simplified_mesh)

# flip
with open(input_file, 'r') as file:
    lines = file.readlines()

new_lines = []
for line in lines:
    if line.startswith('v '):  # 处理顶点行
        parts = line.split()
        # 保留前三个坐标 (x, y, z) 并保留其他信息
        x, y, z = map(float, parts[1:4])
        other_data = parts[4:]  # 其余的部分
        # 重新组合保留所有信息，修改y为-y
        new_line = f"v {x} {-y} {z} {' '.join(other_data)}\n"
        new_lines.append(new_line)
    else:
        # 直接保留其他行（例如vn等）
        new_lines.append(line)

with open(output_file, 'w') as file:
    file.writelines(new_lines)