import math
import time
import random
import signal
import threading
import multiprocessing
import psutil
import cairo
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GLib, Pango, PangoCairo


## Helpers ##

def pretty_bytes(bytecount):

    """
    Return a human readable representation given a size in bytes.
    """

    units = ['Byte', 'Kilobuns', 'Megabuns', 'Gigabuns', 'Terabuns']

    value = bytecount
    for unit in units:
        if value / 1024.0 < 1:
            break

        value /= 1024.0

    return "%.2f %s" % (value, unit)


def render_caption(context, text, x, y, color=None):
    #import pudb; pudb.set_trace()
    context.translate(x, y)
    if color is None:
        color = (1,1,1, 0.6)
    
    context.set_source_rgba(*color)
    
    font = Pango.FontDescription('Orbitron 14')

    layout = PangoCairo.create_layout(context)
    layout.set_font_description(font)
    layout.set_text(text, -1)
    print "size2: ", layout.get_pixel_size()
    size = layout.get_pixel_size()
    context.translate(-size[0] / 2, -size[1] / 2)

    PangoCairo.update_layout(context, layout)
    PangoCairo.show_layout(context, layout)
    context.translate(size[0] / 2, size[1] / 2)
    context.translate(-x, -y)


##class PeriodicCall(threading.Thread):
#
#    """ Periodically forces a window to redraw """
#
#    daemon = True
#    target = None
#    interval = None
#
#    def __init__(self, target, hz):
#        
#        super(PeriodicCall, self).__init__()
#        self.daemon = True
#
#        self.target = target
#        self.interval = 1.0/hz
#
#
#    def run(self):
#
#        while True: # This thread will automatically die with its parent because of the daemon flag
#
#            self.target()
#            time.sleep(self.interval)


class Window(Gtk.Window):

    def __init__(self):

        super(Window, self).__init__()

        self.set_title('hugin')
        self.set_role('hugin')
        self.resize(200, 1080-32)
        self.stick() # show this window on every virtual desktop

        self.set_app_paintable(True)
        self.set_type_hint(Gdk.WindowTypeHint.DOCK)
        self.set_keep_below(True)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual != None and screen.is_composited():
            self.set_visual(visual)



        self.show_all()
        self.move(0, 32) # move apparently must be called after show_all


    @property
    def width(self):

        return self.get_size()[0]


    @property
    def height(self):

        return self.get_size()[1]


class Collector(multiprocessing.Process):

    def __init__(self, queue_update, queue_data):

        super(Collector, self).__init__()
        self.daemon = True
        self.queue_update = queue_update
        self.queue_data = queue_data


    def run(self):

        while True:

            msg = self.queue_update.get(block=True)
            if msg == 'UPDATE':
                self.update()


    def update(self):
        raise NotImplementedError("%s.update not implemented!" % self.__class__.__name__)


class CPUCollector(Collector):

    def update(self):

        aggregate = psutil.cpu_percent(percpu=False)
        percpu = psutil.cpu_percent(percpu=True)

        self.queue_data.put([aggregate, percpu])

        time.sleep(0.1) # according to psutil docs, there should at least be 0.1 seconds between calls to cpu_percent without interval


class MemoryCollector(Collector):

    def update(self):

        data = psutil.virtual_memory()
        self.queue_data.put(data, block=True)


class Monitor(threading.Thread):

    collector_type = Collector

    def __init__(self):

        super(Monitor, self).__init__()
        self.daemon = True

        self.queue_update = multiprocessing.Queue(1)
        self.queue_data = multiprocessing.Queue(1)
        self.collector = self.collector_type(self.queue_update, self.queue_data)
        self.data = []


    def tick(self):
            
        if not self.queue_update.full():
            self.queue_update.put('UPDATE', False)


    def start(self):

        self.collector.start()
        super(Monitor, self).start()


    def run(self):

        while self.collector.is_alive():
            data = self.queue_data.get(block=True) # get new data from the collector as soon as it's available
            self.data = data


    def normalize(self, idx=None):
        raise NotImplementedError("%s.normalize not implemented!" % self.__class__.__name__)


class CPUMonitor(Monitor):

    collector_type = CPUCollector


    def normalized(self, idx=None):

        if len(self.data):
            if isinstance(idx, int):
                return self.data[1][idx] / 100.0
            return self.data[0] / 100.0


class MemoryMonitor(Monitor):

    collector_type = MemoryCollector


    def normalized(self, idx=None):
        if len(self.data):
            return self.data.percent / 100.0


class Gauge(object):

    x = None
    y = None
    width = None
    height = None
    padding = None
    normalize_idx = None
    captions = None

    def __init__(self, x=0, y=0, width=100, height=100, padding=5, normalize_idx=None, captions=None):

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.padding = padding
        self.normalize_idx = normalize_idx
        self.captions = captions if captions else list()


    def update(self, context, value):

        for caption in self.captions:

            if caption.has_key('position'):
                position = caption['position']

            else:
                position = [0, 0]

            render_caption(context, caption['text'], position[0], position[1])


class ArcGauge(Gauge):

    stroke_width = None
    radius = None
    x_center = None
    y_center = None

    def __init__(self, stroke_width=5, **kwargs):

        super(ArcGauge, self).__init__(**kwargs)
        self.stroke_width = stroke_width
        self.radius = (min(self.width, self.height) / 2) - (2 * self.padding) - (self.stroke_width / 2)
        self.x_center = self.x + self.width / 2
        self.y_center = self.y + self.height / 2


    def update(self, context, value):
       
        context.set_line_width(self.stroke_width)
        context.set_line_cap(cairo.LINE_CAP_ROUND)

        context.set_source_rgba(1, 1, 1, 0.1)
        context.arc( # shadow arc
            self.x_center,
            self.y_center,
            self.radius,
            0,
            math.pi * 2
        )
        
        context.stroke()
        

        context.set_source_rgba(0.5, 1, 0, 0.6)

        context.arc(
            self.x_center,
            self.y_center,
            self.radius,
            math.pi / 2,
            math.pi / 2 + math.pi * 2 * value
        )

        context.stroke()

        super(ArcGauge, self).update(context, value)


class Hugin(object):

    window = None
    monitors = None
    gauges = None


    def __init__(self):

        self.window = Window()
        self.window.connect('draw', self.draw)
        
        self.monitors = {}
        self.monitors['cpu'] = CPUMonitor()
        self.monitors['memory'] = MemoryMonitor()

        self.gauges = {}


    def tick(self):

        #print "tick"
        for monitor in self.monitors.itervalues():
            monitor.tick()
        self.window.queue_draw()
        return True # gtk stops executing timeout callbacks if they don't return True


    def draw(self, window, context):

        context.set_operator(cairo.OPERATOR_CLEAR)
        context.paint()
        context.set_operator(cairo.OPERATOR_OVER)

        for source, monitor in self.monitors.iteritems():

            if len(monitor.data) and self.gauges.has_key(source):
                gauges = self.gauges[source]

                for gauge in gauges:
                    gauge.update(context, monitor.normalized(gauge.normalize_idx))


    def start(self):

        for monitor in self.monitors.itervalues():
            monitor.start()

        signal.signal(signal.SIGINT, Gtk.main_quit) # so ctrl+c actually kills hugin
        GLib.timeout_add(1000/3, self.tick)
        Gtk.main()
        print "Thank you for flying with phryk evil mad sciences, LLC. Please come again."


hugin = Hugin()


hugin.gauges['cpu'] = [

    ArcGauge( # aggregate load
        x=0,
        y=0,
        width=hugin.window.width,
        height=150,
        stroke_width=10,
        captions=[
            {
                'text': 'CPU aggregate',
                'position': [100, 80]
            }
        ]
    ), 

    ArcGauge(x=0, y=150, width=hugin.window.width / 2, height=100, normalize_idx=0), 
    ArcGauge(x=hugin.window.width/2, y=150, width=hugin.window.width / 2, height=100, normalize_idx=1), 
    ArcGauge(x=0, y=250, width=hugin.window.width / 2, height=100, normalize_idx=2), 
    ArcGauge(x=hugin.window.width/2, y=250, width=hugin.window.width / 2, height=100, normalize_idx=3) 
]

hugin.gauges['memory'] = [
    ArcGauge(x=0, y=400, width=hugin.window.width, height=hugin.window.width, stroke_width=30)
]

hugin.start()
