add_requires("glfw 3.3.8")

on_load(
  function ()
    function ends_with(str, ending)
      return ending == "" or str:sub(-#ending) == ending
    end
    spirv_files = {}
    for _,src_file in ipairs(os.files(os.projectdir().."/src/*")) do
      if ends_with(src_file, ".vert") or ends_with(src_file, ".frag") or ends_with(src_file, ".comp") then
        print(src_file)
        spirv_files[src_file] = src_file..".spv"
      end
    end
  end
)

rule("glsl")
  set_extensions(".vert", ".frag", ".comp")
  on_build(
    function (target, sourcefile)
      os.run("glslc -o "..target.." "..sourcefile)
    end
  )

for src_file,spirv_file in pairs(spirv_files) do
  target(spirv_file)
    add_files(src_file)
end

target("test")
  set_kind("binary")
  add_files("src/*.cpp", "libs/loguru/loguru.cpp")
  add_includedirs("libs/loguru")
  add_links("glfw")
  if is_mode("debug") then
    add_cxflags("-g3")
  end

  for _,target in pairs(spirv_files) do
    add_deps(target)
  end

