import os
import subprocess

print("=== EGL 系统检查 ===")

# 检查 EGL 库
print("1. 检查 EGL 库:")
result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
egl_libs = [line for line in result.stdout.split('\n') if 'libEGL' in line]
for lib in egl_libs:
    print(f"   {lib}")

# 检查 EGL 供应商配置
print("\n2. 检查 EGL 供应商配置:")
vendor_dir = '/usr/share/glvnd/egl_vendor.d/'
if os.path.exists(vendor_dir):
    for file in os.listdir(vendor_dir):
        print(f"   {vendor_dir}{file}")
        with open(os.path.join(vendor_dir, file), 'r') as f:
            content = f.read()
            if 'nvidia' in content.lower():
                print("    包含 NVIDIA 配置")

# 检查当前环境变量
print("\n3. 相关环境变量:")
egl_vars = [var for var in os.environ if 'EGL' in var or 'GL' in var]
for var in egl_vars:
    print(f"   {var}={os.environ[var]}")

print("\n4. 尝试直接加载 EGL:")
try:
    from OpenGL import EGL
    print("   ✓ 可以通过 PyOpenGL 访问 EGL")
except Exception as e:
    print(f"   ✗ 无法通过 PyOpenGL 访问 EGL: {e}")
