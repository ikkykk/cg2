import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

# 三角形顶点定义
triangle_vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
triangle_vertices[0] = [2.0, 0.0, -2.0]
triangle_vertices[1] = [0.0, 2.0, -2.0]
triangle_vertices[2] = [-2.0, 0.0, -2.0]

# 颜色定义
colors = [
    (1.0, 0.0, 0.0),  # 红色
    (0.0, 1.0, 0.0),  # 绿色
    (0.0, 0.0, 1.0)   # 蓝色
]


def get_model_matrix(angle):
    angle_rad = angle * np.pi / 180.0
    return np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0.0, 0.0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0.0, 0.0],
        [0.0,                0.0,              1.0, 0.0],
        [0.0,                0.0,              0.0, 1.0]
    ])


def get_view_matrix(eye_pos):
    return np.array([
        [1.0, 0.0, 0.0, -eye_pos[0]],
        [0.0, 1.0, 0.0, -eye_pos[1]],
        [0.0, 0.0, 1.0, -eye_pos[2]],
        [0.0, 0.0, 0.0, 1.0]
    ])


def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    eye_fov_rad = eye_fov * np.pi / 180.0
    t = np.tan(eye_fov_rad / 2.0) * zNear
    r = aspect_ratio * t
    l = -r
    b = -t
    n = -zNear
    f = -zFar

    M_persp_to_ortho = np.array([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    M_ortho = np.array([
        [2.0 / (r - l), 0.0, 0.0, -(r + l) / (r - l)],
        [0.0, 2.0 / (t - b), 0.0, -(t + b) / (t - b)],
        [0.0, 0.0, 2.0 / (n - f), -(n + f) / (n - f)],
        [0.0, 0.0, 0.0, 1.0]
    ])

    return M_ortho @ M_persp_to_ortho


def main():
    gui = ti.GUI('旋转与变换', res=(700, 700))
    angle = 0.0
    eye_pos = np.array([0.0, 0.0, 5.0])

    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key == ti.GUI.ESCAPE:
                exit()
            elif e.key == 'a' or e.key == 'A':
                angle += 10.0
            elif e.key == 'd' or e.key == 'D':
                angle -= 10.0

        M_model = get_model_matrix(angle)
        M_view = get_view_matrix(eye_pos)
        M_proj = get_projection_matrix(45.0, 1.0, 0.1, 50.0)
        M_mvp = M_proj @ M_view @ M_model

        transformed_vertices = []
        for i in range(3):
            v = np.array([triangle_vertices[i][0], triangle_vertices[i][1], triangle_vertices[i][2], 1.0])
            v_transformed = M_mvp @ v
            v_transformed /= v_transformed[3]
            transformed_vertices.append([
                (v_transformed[0] + 1.0) * 0.5,
                (v_transformed[1] + 1.0) * 0.5
            ])

        gui.lines(begin=np.array([transformed_vertices[0]]),
                  end=np.array([transformed_vertices[1]]),
                  color=0xff0000, radius=2)
        gui.lines(begin=np.array([transformed_vertices[1]]),
                  end=np.array([transformed_vertices[2]]),
                  color=0x00ff00, radius=2)
        gui.lines(begin=np.array([transformed_vertices[2]]),
                  end=np.array([transformed_vertices[0]]),
                  color=0x0000ff, radius=2)

        gui.show()


if __name__ == '__main__':
    main()
