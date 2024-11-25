import sys
sys.dont_write_bytecode = True
import os, subprocess

if __name__ == '__main__':
  print("Compiling shader")
  script_directory = os.path.dirname(os.path.realpath(__file__))
  os.chdir(script_directory)
  if not os.path.exists("bin/"):
    os.mkdir("bin/")
  dirs = [d for d in os.listdir(".") if not d.endswith('.glsl') and not d.endswith('.py') and os.path.isfile(d)]
  for d in dirs:
    try:
      print('"{}"'.format(d))
      glslc_args = "--target-env=vulkan1.2 -O -o bin/{0}.spv {0}".format(d)
      if os.name == 'nt':
        subprocess.run(f".\\..\\dependencies\\glslc.exe {glslc_args}", shell=True, check=True)
      else:
        subprocess.run(f"glslc {glslc_args}", shell=True, check=True)
    except subprocess.CalledProcessError as e:
      print("ERROR")
      exit(1)
  print("Shader compiled!")
