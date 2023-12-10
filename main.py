import ctypes
import os
import sys

sys.path.append('..')

import pyglet
from pyglet.gl import *

from pywavefront import visualization
import pywavefront

# Create absolute path from this module
file_abspath = os.path.join(os.path.dirname(__file__), 'untitled.obj')

rotation = 0
snapshot_counter = 0
meshes = pywavefront.Wavefront(file_abspath)
window = pyglet.window.Window(resizable=True)
lightfv = ctypes.c_float * 4


@window.event
def on_resize(width, height):
    viewport_width, viewport_height = window.get_framebuffer_size()
    glViewport(0, 0, viewport_width, viewport_height)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., float(width)/height, 1., 100.)
    glMatrixMode(GL_MODELVIEW)
    return True


@window.event
def on_draw():
    global snapshot_counter
    window.clear()
    glLoadIdentity()

    glLightfv(GL_LIGHT0, GL_POSITION, lightfv(-1.0, 1.0, 1.0, 0.0))
    glEnable(GL_LIGHT0)

    glTranslated(0.0, 0.0, -5.0)
    glRotatef(rotation, 0.0, 1.0, 0.0)
    glRotatef(-25.0, 1.0, 0.0, 0.0)
    glRotatef(45.0, 0.0, 0.0, 1.0) 

    glEnable(GL_LIGHTING)
    visualization.draw(meshes)

    # Save the current frame as a JPG image
    filename = f"snapshot_{snapshot_counter}.jpg"
    pyglet.image.get_buffer_manager().get_color_buffer().save("Img/"+filename)
    print(f"Saved snapshot to {filename}")
    snapshot_counter += 1

def update(dt):
    global rotation
    rotation += 90.0 * dt

    if rotation > 720.0:
        rotation = 0.0


pyglet.clock.schedule(update)
pyglet.app.run()