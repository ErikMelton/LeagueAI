import bpy
from math import *

# light
lamp = bpy.data.objects['Lamp']
lamp.location[0] = 0
lamp.location[1] = -4
lamp.location[2] = 6
lamp.data.type = 'SUN'
lamp.data.energy = 2.19
lamp.data.use_specular = False
lamp.data.shadow_method = 'NOSHADOW'

# Camera
cam = bpy.data.objects['Camera']
cam.location[0] = 0
cam.location[1] = 0
cam.location[2] = 16
cam.rotation_euler[0] = 0
cam.rotation_euler[1] = 0
cam.rotation_euler[2] = 0

bpy.context.scene.render.use_shadows = False
bpy.context.scene.render.alpha_mode = 'TRANSPARENT'

step_count = 32
dist_count = 10

for dist in range(0, dist_count):
    cam.location[2] = cam.location[2] + 1
    
    for step in range(0, step_count):
        cam.rotation_euler[2] = radians(step * (360.0 / step_count))

        bpy.data.scenes["Scene"].render.filepath = 'D:\\code/LeagueAI/data/characters/minion_ranged_enemy/order%d_%d.jpg' % (step, dist)
        bpy.ops.render.render( write_still=True )
