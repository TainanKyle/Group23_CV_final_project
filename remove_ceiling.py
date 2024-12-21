def remove_ceiling(input_file, output_file, y_threshold=0.95):
    """
    從 .obj 檔案中移除 y 值大於給定百分比的點（用來去除天花板）。

    Args:
        input_file (str): 輸入的 .obj 檔案路徑。
        output_file (str): 輸出的 .obj 檔案路徑。
        y_threshold (float): y 值的百分比閾值（0~1）。
    """
    vertices = []
    faces = []
    y_values = []

    # 讀取 .obj 檔案
    with open(input_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('v '):  # 頂點
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append((x, y, z))
                y_values.append(y)
            elif line.startswith('f '):  # 面
                faces.append(line.strip())

    # 計算 y 值的 95% 閾值
    y_threshold_value = sorted(y_values)[int(len(y_values) * y_threshold)]

    # 過濾頂點
    filtered_vertices = []
    vertex_map = {}  # 用來記錄過濾後的頂點索引對應
    for idx, (x, y, z) in enumerate(vertices):
        if y <= y_threshold_value:
            vertex_map[idx + 1] = len(filtered_vertices) + 1  # .obj 的索引從 1 開始
            filtered_vertices.append((x, y, z))

    # 更新面（過濾掉已移除的頂點）
    filtered_faces = []
    for face in faces:
        indices = [int(i.split('/')[0]) for i in face.split()[1:]]
        if all(index in vertex_map for index in indices):  # 確保面所有的頂點都在
            remapped_indices = [vertex_map[index] for index in indices]
            filtered_faces.append(f"f {' '.join(map(str, remapped_indices))}")

    # 寫入新 .obj 檔案
    with open(output_file, 'w') as file:
        for x, y, z in filtered_vertices:
            file.write(f"v {x} {y} {z}\n")
        for face in filtered_faces:
            file.write(face + '\n')

# 使用範例
input_obj = "/home/gpl_homee/indoor_scene/SceneTex/data/scenes/room_21/meshes/scene.obj"  # 替換成你的輸入檔案路徑
output_obj = "scene.obj"  # 替換成你想要的輸出檔案路徑
remove_ceiling(input_obj, output_obj, y_threshold=0.92)
