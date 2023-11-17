add_requires("glfw 3.3.8")

target("test")
  set_kind("binary")
  add_files("src/*.cpp", "libs/loguru/loguru.cpp")
  add_includedirs("libs/loguru")
  add_links("glfw")
  if is_mode("debug") then
    add_cxflags("-g3")
  end

