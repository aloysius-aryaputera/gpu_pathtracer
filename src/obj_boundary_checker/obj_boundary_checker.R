OBJ_FILE = '/home/aloysius/Documents/GitHub/gpu_pathtracer/src/obj/dragon.obj'

SX = .45
SY = .45
SZ = .45

TX = 0
TY = 0
TZ = 0

obj_df = read.table(OBJ_FILE)
names(obj_df) = c('code', 'x', 'y', 'z')
  obj_df = obj_df[which(obj_df$code == 'v'),]

obj_df$x = SX * obj_df$x
obj_df$y = SY * obj_df$y
obj_df$z = SZ * obj_df$z

obj_df$x = TX + obj_df$x
obj_df$y = TY + obj_df$y
obj_df$z = TZ + obj_df$z

print(paste0("x min = ", min(obj_df$x)))
print(paste0("x max = ", max(obj_df$x)))
print('')
print(paste0("y min = ", min(obj_df$y)))
print(paste0("y max = ", max(obj_df$y)))
print('')
print(paste0("z min = ", min(obj_df$z)))
print(paste0("z max = ", max(obj_df$z)))