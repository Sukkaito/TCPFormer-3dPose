import sys

with open('local_pose_3d_server.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(201, 299):
    if lines[i].strip():
        lines[i] = '    ' + lines[i]
    else:
        lines[i] = '\n'

with open('local_pose_3d_server.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("Indentation fixed.")
