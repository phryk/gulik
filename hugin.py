import time
import random
import threading
import psutil
import cairo
#import Gtk # for Gtk.gdk.get_default_root_window
import gi
gi.require_version('Gtk', '3.0')

from gi.repository import Gtk, Gdk


#window = Gtk.gdk.get_default_root_window()
#context = window.cairo_create()

class Monitor(Gtk.Window):

    def __init__(self):

        super(Monitor, self).__init__()

        self.set_title('hugin')
        self.set_role('hugin')
        self.resize(200, 1080-36)
        #self.set_resizable(False)
        #self.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
        self.move(100, 0)

        self.set_app_paintable(True)
        self.set_type_hint(Gdk.WindowTypeHint.DOCK)
        self.set_keep_below(True)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual != None and screen.is_composited():
            self.set_visual(visual)


        self.connect('draw', self.tick)

        self.connect('delete-event', Gtk.main_quit)
        self.show_all()


    def tick(self, wid, cr):

        #import pudb; pudb.set_trace()

        cr.set_operator(cairo.OPERATOR_CLEAR)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        cr.set_source_rgba(1, 0, 0, 0.4)
        cr.rectangle(0, 0, self.width, self.height)
        cr.fill()

        #print "tick!"


    @property
    def width(self):

        return self.get_size()[0]


    @property
    def height(self):

        return self.get_size()[1]


class Refresher(threading.Thread):

    window = None
    interval = None

    def __init__(self, window, interval):
        
        super(Refresher, self).__init__()
        self.daemon = True

        self.window = window
        self.interval = interval



    def run(self):

        while True:
            #print "florb"
            self.window.queue_draw()
            time.sleep(self.interval)


m = Monitor()
t = Refresher(m, 0.1)
t.start()

Gtk.main()
