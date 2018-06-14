import math
import time
import random
import signal
import collections
import threading
import multiprocessing
import psutil
import cairo
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('PangoCairo', '1.0') # not sure if want
from gi.repository import Gtk, Gdk, GLib, Pango, PangoCairo


# CONFIG: TODO: Move into its own file, obvsly
CONFIG_FPS = 3
CONFIG_COLORS = {
    'window_background': (0,0,0, 0.6),
    'gauge_background': (1,1,1, 0.1),
    'highlight': (0.5, 1, 0, 0.6),
}

## Helpers ##

def pretty_bytes(bytecount):

    """
    Return a human readable representation given a size in bytes.
    """

    units = ['Byte', 'kbyte', 'Mbyte', 'Gbyte', 'Tbyte']

    value = bytecount
    for unit in units:
        if value / 1024.0 < 1:
            break

        value /= 1024.0

    return "%.2f %s" % (value, unit)


def pretty_bits(bytecount):

    """
    Return a human readable representation given a size in bytes.
    """

    units = ['bit', 'kbit', 'Mbit', 'Gbit', 'Tbit']

    value = bytecount * 8 # bytes to bits
    for unit in units:
        if value / 1024.0 < 1:
            break

        value /= 1024.0

    return "%.2f %s" % (value, unit)


def alignment_offset(align, size):

    x_align, y_align = align.split('_')

    if x_align == 'left':
        x_offset = 0
    elif x_align == 'center':
        x_offset = -size[0] / 2
    elif x_align == 'right':
        x_offset = -size[0]
    else:
        raise ValueError("unknown horizontal alignment: '%s', must be one of: left, center, right" % x_align)

    if y_align == 'top':
        y_offset = 0
    elif y_align == 'center':
        y_offset = -size[1] / 2
    elif y_align == 'bottom':
        y_offset = -size[1]
    else:
        raise ValueError("unknown horizontal alignment: '%s', must be one of: top, center, bottom" % y_align)


    return (x_offset, y_offset)


def render_caption(context, text, x, y, align=None, color=None, font_size=None):
    
    if align is None:
        align = 'left_top'
    
    if color is None:
        color = (1,1,1, 0.6)
    
    context.set_source_rgba(*color)

    if font_size is None:
        font_size = 14

    font = Pango.FontDescription('Orbitron %d' % font_size)

    layout = PangoCairo.create_layout(context)
    layout.set_font_description(font)
    layout.set_text(text, -1)
    
    size = layout.get_pixel_size()

    x_offset, y_offset = alignment_offset(align, size)
    
    context.translate(x + x_offset, y + y_offset)

    PangoCairo.update_layout(context, layout)
    PangoCairo.show_layout(context, layout)

    context.translate(-x - x_offset, -y - y_offset)


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
        self.queue_data.put(
            {
                'aggregate': aggregate, 
                'percpu': percpu
            }
        )
        
        time.sleep(0.1) # according to psutil docs, there should at least be 0.1 seconds between calls to cpu_percent without interval


class MemoryCollector(Collector):

    def update(self):

        self.queue_data.put(
            psutil.virtual_memory(),
            block=True
        )


class NetworkCollector(Collector):


    def update(self):

        counters = psutil.net_io_counters(pernic=True, nowrap=True)
        sockets = psutil.net_connections(kind='all')
        stats = psutil.net_if_stats()
        self.queue_data.put([counters, sockets, stats])

        psutil.net_io_counters.cache_clear()
        #time.sleep(0.1)


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


    def normalized(self, idx=None):
        raise NotImplementedError("%s.normalize not implemented!" % self.__class__.__name__)


    def caption(self, fmt, idx=None):
        raise NotImplementedError("%s.caption not implemented!" % self.__class__.__name__)


class CPUMonitor(Monitor):

    collector_type = CPUCollector

    def normalized(self, idx=None):

        if isinstance(idx, int):
            return self.data['percpu'][idx] / 100.0
        return self.data['aggregate'] / 100.0


    def caption(self, fmt, idx=None):

        if isinstance(idx, int):
            return fmt.format(self.data['percpu'][idx])

        return fmt.format(self.data['aggregate'])


class MemoryMonitor(Monitor):

    collector_type = MemoryCollector

    def normalized(self, idx=None):
        if len(self.data):
            return self.data.percent / 100.0


    def caption(self, fmt, idx=None):
        return fmt.format(**self.data._asdict())


class NetworkMonitor(Monitor):

    collector_type = NetworkCollector
    interfaces = None


    def __init__(self):

        super(NetworkMonitor, self).__init__()

        self.interfaces = collections.OrderedDict()

        for if_name in psutil.net_if_stats().keys():
                self.interfaces[if_name] = {}
                self.interfaces[if_name]['counts'] = {}
                for key in ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv', 'errin', 'errout', 'dropin', 'dropout']:
                    self.interfaces[if_name]['counts'][key] = collections.deque([], CONFIG_FPS) # max size equal fps means this holds data of only the last second


    def run(self):

        while self.collector.is_alive():
            data = self.queue_data.get(block=True) # get new data from the collector as soon as it's available
            self.data = data
            for if_name, if_info in self.interfaces.iteritems():
                for key, deque in if_info['counts'].iteritems():
                    deque.append(self.data[0][if_name]._asdict()[key])


    def count_sec(self, interface, key):

        """
            get a specified count for a given interface as calculated for the last second
            EXAMPLE: self.count_sec('eth0', 'bytes_sent') will return count of bytes sent in the last second
        """

        deque = self.interfaces[interface]['counts'][key]
        return deque[-1] - deque[0] # last (most recent) minus first (oldest) item


    def normalized(self, idx=None):
        if len(self.data):

            if self.data[2][idx].speed:
                link_quality = float(self.data[2][idx].speed * 1024**2)
            else:
                link_quality = float(100 * 1024**2)

            return (self.count_sec(idx, 'bytes_recv') * 8) / link_quality


    def caption(self, fmt, idx=None):
 
        if self.interfaces.has_key(idx) and self.interfaces[idx].has_key('counts'):
            data = {}
            for k in self.interfaces[idx]['counts'].keys():

                data[k] = self.count_sec(idx, k)
                if k.startswith('bytes'):
                    data[k] = pretty_bits(data[k])


            return fmt.format(**data)


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


    @property
    def inner_width(self):
        return self.width - 2 * self.padding


    @property
    def inner_height(self):
        return self.height - 2 * self.padding


    def update(self, context, monitor):

        """
        parameters:
            context: cairo context of the window
            monitor: A Monitor object holding relevant data and helper functions
        """

        for caption in self.captions:

            if caption.has_key('position'):


                if isinstance(caption['position'], basestring):
                    # handle alignment-style strings like "center_bottom"
                    position = [-x for x in alignment_offset(caption['position'], (self.width - 2 * self.padding, self.height - 2 * self.padding))]

                else:
                    position = caption['position']

            else:
                position = [0, 0]

            position = [position[0] + self.x + self.padding, position[1] + self.y + self.padding]


            caption_text = monitor.caption(caption['text'], idx=self.normalize_idx)

            render_caption(context, caption_text, position[0], position[1], align=caption.get('align', None), font_size=caption.get('font_size', None))


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


    def update(self, context, monitor):

        context.set_line_width(self.stroke_width)
        #context.set_line_cap(cairo.LINE_CAP_ROUND)
        context.set_line_cap(cairo.LINE_CAP_BUTT)

        context.set_source_rgba(*CONFIG_COLORS['gauge_background'])
        context.arc( # shadow arc
            self.x_center,
            self.y_center,
            self.radius,
            0,
            math.pi * 2
        )
        
        context.stroke()
        

        context.set_source_rgba(*CONFIG_COLORS['highlight'])

        context.arc(
            self.x_center,
            self.y_center,
            self.radius,
            math.pi / 2,
            math.pi / 2 + math.pi * 2 * monitor.normalized(self.normalize_idx)
        )

        context.stroke()

        super(ArcGauge, self).update(context, monitor)


class PlotGauge(Gauge):

    points = None
    num_points = None
    autoscale = None

    def __init__(self, num_points=None, autoscale=True, **kwargs):

        super(PlotGauge, self).__init__(**kwargs)

        if num_points:
            self.num_points = num_points
        else:
            self.num_points = self.inner_width / 8 + 1

        self.points = collections.deque([], self.num_points)

        self.autoscale = autoscale


    @property
    def points_scaled(self):
        
        scale_factor = max(self.points)

        if scale_factor == 0.0:
            print "zero"
            return [0.0 for _ in range(0, len(self.points))]

        r = []
        for amplitude in self.points:
            r.append(amplitude / scale_factor)

        return r


    def update(self, context, monitor):
            
        self.points.append(monitor.normalized(self.normalize_idx))
        
        context.set_line_width(2)
        context.set_source_rgba(*CONFIG_COLORS['gauge_background'])
        context.rectangle(self.x + self.padding, self.y + self.padding, self.inner_width, self.inner_height)
        context.fill()

        coords = []
        if self.autoscale:
            points = self.points_scaled
        else:
            points = self.points

        for idx, amplitude in enumerate(points):
            coords.append((
                idx * 8 + self.padding,
                self.y + self.padding + self.inner_height - (self.inner_height * amplitude)
            ))
      
        context.set_source_rgba(*CONFIG_COLORS['highlight'])
        context.set_line_width(2)
        #context.set_line_cap(cairo.LINE_CAP_BUTT)
        # draw lines
        for idx, (x, y) in enumerate(coords):
            if idx == 0:
                context.move_to(x, y)
            else:
                context.line_to(x, y)

        context.stroke()

        # place points
        for (x, y) in coords:

            context.set_source_rgba(*CONFIG_COLORS['highlight'])

            context.arc(
                x,
                y,
                2,
                0,
                2 * math.pi
            )
        
            context.fill()

        super(PlotGauge, self).update(context, monitor)


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
        self.monitors['network'] = NetworkMonitor()

        self.gauges = {}


    def tick(self):

        for monitor in self.monitors.itervalues():
            monitor.tick()
        self.window.queue_draw()
        return True # gtk stops executing timeout callbacks if they don't return True


    def draw(self, window, context):

        context.set_operator(cairo.OPERATOR_CLEAR)
        context.paint()
        context.set_operator(cairo.OPERATOR_OVER)

        context.set_source_rgba(*CONFIG_COLORS['window_background'])
        context.rectangle(0, 0, self.window.width, self.window.height)
        context.fill()

        for source, monitor in self.monitors.iteritems():

            if len(monitor.data) and self.gauges.has_key(source):
                gauges = self.gauges[source]

                for gauge in gauges:
                    gauge.update(context, monitor)


    def start(self):

        for monitor in self.monitors.itervalues():
            monitor.start()

        signal.signal(signal.SIGINT, Gtk.main_quit) # so ctrl+c actually kills hugin
        GLib.timeout_add(1000/CONFIG_FPS, self.tick)
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
                'text': '{:.1f}%',
                'position': 'center_center',
                'align': 'center_center'
            }
        ]
    ), 

    ArcGauge(
        x=0,
        y=150,
        width=hugin.window.width / 2,
        height=100,
        normalize_idx=0,
        captions=[
            {
                'text': '{:.1f}%',
                'position': 'center_center',
                'align': 'center_center',
                'font_size': 10,
            }
        ]
    ),

    ArcGauge(
        x=hugin.window.width / 2,
        y=150,
        width=hugin.window.width / 2,
        height=100,
        normalize_idx=1,
        captions=[
            {
                'text': '{:.1f}%',
                'position': 'center_center',
                'align': 'center_center',
                'font_size': 10,
            }
        ]
    ),

    ArcGauge(
        x=0,
        y=250,
        width=hugin.window.width / 2,
        height=100,
        normalize_idx=2,
        captions=[
            {
                'text': '{:.1f}%',
                'position': 'center_center',
                'align': 'center_center',
                'font_size': 10,
            }
        ]
    ),

    ArcGauge(
        x=hugin.window.width / 2,
        y=250,
        width=hugin.window.width / 2,
        height=100,
        normalize_idx=3,
        captions=[
            {
                'text': '{:.1f}%',
                'position': 'center_center',
                'align': 'center_center',
                'font_size': 10,
            }
        ]
    ), 
]

hugin.gauges['memory'] = [
    ArcGauge(
        x=0,
        y=400,
        width=hugin.window.width,
        height=hugin.window.width,
        stroke_width=30,
        captions=[
            {
                'text': '{percent:.1f}%',
                'position': 'center_center',
                'align': 'center_center'
            }
        ]
    )
]

hugin.gauges['network'] = [
    ArcGauge(
        x=0,
        y=600,
        width=hugin.window.width,
        height=hugin.window.width,
        normalize_idx='re0',
        captions=[
            {
                'text': '{bytes_recv}/s',
                'position': 'center_center',
                'align': 'center_center',
            }
        ]
    ),

    PlotGauge(
        x=0,
        y=800,
        width=hugin.window.width,
        height=100,
        padding=15,
        normalize_idx='re0'
    )
]

hugin.start()
