library(readtext)

#OBJ_FILE = '/home/aloysius/Documents/GitHub/gpu_pathtracer/src/obj/camel.obj'
OBJ_FILE = '/home/aloysius/Downloads/statue.obj'
#OBJ_FILE = '/home/aloysius/Desktop/car.obj'

raw_text = readtext(OBJ_FILE)
lines = strsplit(raw_text$text, "\n")

tmp_file = "tmp.txt"
file.create(tmp_file)
for(line in lines[[1]]) {
  sub_line = strsplit(line, " ")[[1]]
  if (length(sub_line)) {
    if (sub_line[1] == "v") {
      write(line, tmp_file, append=TRUE)
    }
  }
}

SX = .4
SY = .4
SZ = .4

TX = 0
TY = 0.8662064
TZ = 0

obj_df = read.table(tmp_file)
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

file.remove(tmp_file)