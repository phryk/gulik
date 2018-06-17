#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import math
import time
import random
import signal
import collections
import threading
import multiprocessing
import psutil
import colorsys
import cairo
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('PangoCairo', '1.0') # not sure if want
from gi.repository import Gtk, Gdk, GLib, Pango, PangoCairo

## Helpers ##

class Color(object):

    red = None
    green = None
    blue = None

    hue = None
    saturation = None
    value = None

    alpha = None


    def __init__(self, red=None, green=None, blue=None, alpha=None, hue=None, saturation=None, value=None):

        rgb_passed = bool(red)|bool(green)|bool(blue)
        hsv_passed = bool(hue)|bool(saturation)|bool(value)

        if not alpha:
            alpha = 0.0

        if rgb_passed and hsv_passed:
            raise ValueError("Color can't be initialized with RGB and HSV at the same time.")

        elif hsv_passed:

            if not hue:
                hue = 0.0
            if not saturation:
                saturation = 0.0
            if not value:
                value = 0.0

            super(Color, self).__setattr__('hue', hue)
            super(Color, self).__setattr__('saturation', saturation)
            super(Color, self).__setattr__('value', value)
            self._update_rgb()

        else:

            if not red:
                red = 0
            if not green:
                green = 0
            if not blue:
                blue = 0

            super(Color, self).__setattr__('red', red)
            super(Color, self).__setattr__('green', green)
            super(Color, self).__setattr__('blue', blue)
            self._update_hsv()

        super(Color, self).__setattr__('alpha', alpha)


    def __setattr__(self, key, value):

        if key in ('red', 'green', 'blue'):
            if value > 1.0:
                value = value % 1.0
            super(Color, self).__setattr__(key, value)
            self._update_hsv()

        elif key in ('hue', 'saturation', 'value'):
            if key == 'hue' and (value >= 360.0 or value < 0):
                value = value % 360.0
            elif key != 'hue' and value > 1.0:
                value = 1.0
            super(Color, self).__setattr__(key, value) 
            self._update_rgb()

        else:
            if key == 'alpha' and value > 1.0: # TODO: Might this be more fitting in another place?
                value = 1.0

            super(Color, self).__setattr__(key, value)


    def __repr__(self):

        return '<%s: red %f, green %f, blue %f, hue %f, saturation %f, value %f, alpha %f>' % (
                self.__class__.__name__,
                self.red,
                self.green,
                self.blue,
                self.hue,
                self.saturation,
                self.value,
                self.alpha
            )


    def clone(self):
        return Color(red=self.red, green=self.green, blue=self.blue, alpha=self.alpha)


    def blend(self, other, mode='normal'):

        if self.alpha != 1.0: # no clue how to blend with a translucent bottom layer
            self.red = self.red * self.alpha
            self.green = self.green * self.alpha
            self.blue = self.blue * self.alpha

            self.alpha = 1.0

        if mode == 'normal':
            own_influence = 1.0 - other.alpha
            self.red = (self.red * own_influence) + (other.red * other.alpha)
            self.green = (self.green * own_influence) + (other.green * other.alpha)
            self.blue = (self.blue * own_influence) + (other.blue * other.alpha)


    def lighten(self, other):

        if isinstance(other, int) or isinstance(other, float):
            other = Color(red=other, green=other, blue=other, alpha=1.0)

        if self.alpha != 1.0:
            self.red = self.red * self.alpha
            self.green = self.green * self.alpha
            self.blue = self.blue * self.alpha

            self.alpha = 1.0

        red = self.red + (other.red * other.alpha)
        green = self.green + (other.green * other.alpha)
        blue = self.blue + (other.blue * other.alpha)

        if red > 1.0:
            red = 1.0

        if green > 1.0:
            green = 1.0

        if blue > 1.0:
            blue = 1.0

        self.red = red
        self.green = green
        self.blue = blue


    def darken(self, other):

        if isinstance(other, int) or isinstance(other, float):
            other = Color(red=other, green=other, blue=other, alpha=1.0)

        red = self.red - other.red
        green = self.green - other.green
        blue = self.blue - other.blue

        if red < 0:
            red = 0

        if green < 0:
            green = 0

        if blue < 0:
            blue = 0

        self.red = red
        self.green = green
        self.blue = blue


    def tuple_rgb(self):
        """ return color (without alpha) as tuple, channels being float 0.0-1.0 """
        return (self.red, self.green, self.blue)
    
    
    def tuple_rgba(self):
        """ return color (*with* alpha) as tuple, channels being float 0.0-1.0 """
        return (self.red, self.green, self.blue, self.alpha)


    def _update_hsv(self):

        hue, saturation, value = colorsys.rgb_to_hsv(self.red, self.green, self.blue)
        super(Color, self).__setattr__('hue', hue * 360.0)
        super(Color, self).__setattr__('saturation', saturation)
        super(Color, self).__setattr__('value', value)


    def _update_rgb(self):

        red, green, blue = colorsys.hsv_to_rgb(self.hue / 360.0, self.saturation, self.value)
        super(Color, self).__setattr__('red', red)
        super(Color, self).__setattr__('green', green)
        super(Color, self).__setattr__('blue', blue)


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

# CONFIG: TODO: Move into its own file, obvsly
CONFIG_FPS = 3
CONFIG_COLORS = {
    'window_background': Color(0,0,0, 0.6),
    'gauge_background': Color(1,1,1, 0.1),
    'highlight': Color(0.5, 1, 0, 0.6),
}

## HUGIN ##

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

        count = psutil.cpu_count()
        aggregate = psutil.cpu_percent(percpu=False)
        percpu = psutil.cpu_percent(percpu=True)
        self.queue_data.put(
            {
                'count': count,
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

        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        counters = psutil.net_io_counters(pernic=True, nowrap=True)
        connections = psutil.net_connections(kind='all')

        self.queue_data.put({
            'stats': stats,
            'addrs': addrs,
            'counters': counters,
            'connections': connections,
        })

        #psutil.net_io_counters.cache_clear()


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


    def normalized(self, address=None):
        raise NotImplementedError("%s.normalize not implemented!" % self.__class__.__name__)


    def caption(self, fmt):
        raise NotImplementedError("%s.caption not implemented!" % self.__class__.__name__)


class CPUMonitor(Monitor):

    collector_type = CPUCollector

    def normalized(self, address=None):

        if isinstance(address, int):
            return self.data['percpu'][address] / 100.0
        return self.data['aggregate'] / 100.0


    def caption(self, fmt):
        
        data = {}
        data['count'] = self.data['count']
        data['aggregate'] = self.data['aggregate']
        for idx, perc in enumerate(self.data['percpu']):
            data['core_%d' % idx] = perc

        return fmt.format(**data)

class MemoryMonitor(Monitor):

    collector_type = MemoryCollector

    def normalized(self, address=None):
        if len(self.data):
            return self.data.percent / 100.0


    def caption(self, fmt):

        data = {}
        for k, v in self.data._asdict().iteritems():
            if k != 'percent':
                v = pretty_bytes(v)

            data[k] = v

        return fmt.format(**data)


class NetworkMonitor(Monitor):

    collector_type = NetworkCollector
    interfaces = None


    def __init__(self):

        super(NetworkMonitor, self).__init__()

        self.interfaces = collections.OrderedDict()

        if CONFIG_FPS < 2:
            deque_len = 2
        else:
            deque_len = CONFIG_FPS
        
        for if_name in psutil.net_if_stats().keys():
                self.interfaces[if_name] = {}
                self.interfaces[if_name]['addrs'] = {}
                self.interfaces[if_name]['stats'] = {}
                self.interfaces[if_name]['counters'] = {}
                for key in ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv', 'errin', 'errout', 'dropin', 'dropout']:
                    self.interfaces[if_name]['counters'][key] = collections.deque([], deque_len) # max size equal fps means this holds data of only the last second


    def run(self):

        while self.collector.is_alive():
            data = self.queue_data.get(block=True) # get new data from the collector as soon as it's available
            self.data = data
            for if_name, if_info in self.interfaces.iteritems():
                
                for key, deque in if_info['counters'].iteritems():
                    deque.append(self.data['counters'][if_name]._asdict()[key])
                
                self.interfaces[if_name]['stats'] = self.data['stats'][if_name]._asdict()
                self.interfaces[if_name]['addrs'] = self.data['addrs'][if_name]


    def count_sec(self, interface, key):

        """
            get a specified count for a given interface as calculated for the last second
            EXAMPLE: self.count_sec('eth0', 'bytes_sent') will return count of bytes sent in the last second
        """

        deque = self.interfaces[interface]['counters'][key]
        
        if CONFIG_FPS < 2:
            return (deque[-1] - deque[0]) / CONFIG_FPS # fps < 1 means data covers 1/fps seconds

        else:
            return deque[-1] - deque[0] # last (most recent) minus first (oldest) item


    def normalized(self, address=None):

        if_name, key = address.split('.')

        if len(self.data):

            if self.interfaces[if_name]['stats']['speed']:
                link_quality = float(self.interfaces[if_name]['stats']['speed'] * 1024**2)
            else: # speed == 0 means it couldn't be determined, fall back to 100Mbit/s
                link_quality = float(100 * 1024**2)

            return (self.count_sec(if_name, key) * 8) / link_quality


    def caption(self, fmt):
        
        data = {}

        for if_name in self.interfaces.keys():

            data[if_name] = {}
            data[if_name]['addrs'] = self.interfaces[if_name]['addrs']
            data[if_name]['stats'] = self.interfaces[if_name]['stats']

            data[if_name]['counters'] = {}
            for k in self.interfaces[if_name]['counters'].keys():

                data[if_name]['counters'][k] = self.count_sec(if_name, k)
                if k.startswith('bytes'):
                    data[if_name]['counters'][k] = pretty_bits(data[if_name]['counters'][k])

        return fmt.format(**data)


class Gauge(object):

    x = None
    y = None
    width = None
    height = None
    padding = None
    address = None
    captions = None
    colors = None

    def __init__(self, x=0, y=0, width=100, height=100, padding=5, address=None, captions=None, foreground=None, background=None):

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.padding = padding
        self.address = address
        self.captions = captions if captions else list()

        self.colors = {}

        if foreground is None:
            self.colors['foreground'] = CONFIG_COLORS['highlight']
        else:
            self.colors['foreground'] = foreground

        if background is None:
            self.colors['background'] = CONFIG_COLORS['gauge_background']
        else:
            self.colors['background'] = background


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


            caption_text = monitor.caption(caption['text'])

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
        context.set_line_cap(cairo.LINE_CAP_BUTT)

        context.set_source_rgba(*self.colors['background'].tuple_rgba())
        context.arc( # shadow arc
            self.x_center,
            self.y_center,
            self.radius,
            0,
            math.pi * 2
        )
        
        context.stroke()

        context.set_source_rgba(*self.colors['foreground'].tuple_rgba())
        context.arc(
            self.x_center,
            self.y_center,
            self.radius,
            math.pi / 2,
            math.pi / 2 + math.pi * 2 * monitor.normalized(self.address)
        )

        context.stroke()

        super(ArcGauge, self).update(context, monitor)


class DualArcGauge(ArcGauge):

    def update(self, context, monitor):

        context.set_line_width(self.stroke_width)
        context.set_line_cap(cairo.LINE_CAP_BUTT)

        context.set_source_rgba(*self.colors['background'].tuple_rgba())
        context.arc( # shadow arc
            self.x_center,
            self.y_center,
            self.radius,
            0,
            math.pi * 2
        )
        context.stroke()


        context.set_source_rgba(*self.colors['foreground'].tuple_rgba())
        context.arc(
            self.x_center,
            self.y_center,
            self.radius,
            math.pi / 2,
            math.pi / 2 + math.pi * monitor.normalized(self.address[0])
        )
        context.stroke()

        context.set_source_rgba(1,0,0.5, 0.6)
        context.arc_negative(
            self.x_center,
            self.y_center,
            self.radius,
            math.pi / 2,
            math.pi / 2 - math.pi * monitor.normalized(self.address[1])
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

        self.colors['grid_major'] = self.colors['foreground'].clone()
        self.colors['grid_major'].alpha *= 0.8

        self.colors['grid_minor'] = self.colors['background'].clone()
        self.colors['grid_minor'].alpha *= 0.8

        self.colors['grid_milli'] = self.colors['background'].clone()
        self.colors['grid_milli'].alpha *= 0.4


    def get_scale_factor(self):
        p = max(self.points)
        if p:
            return 1.0 / p
        return 0.0


    def get_points_scaled(self):
       
        scale_factor = self.get_scale_factor()
        if scale_factor == 0.0:
            return [0.0 for _ in range(0, len(self.points))]

        r = []
        for amplitude in self.points:
            r.append(amplitude * scale_factor)

        return r


    def draw_grid(self, context, monitor):

        scale_factor = self.get_scale_factor()

        context.set_line_width(1)
        context.set_source_rgba(*self.colors['grid_minor'].tuple_rgba())
        #context.set_dash([1,1])

        for x in range(self.x + self.padding, self.x + self.padding + self.inner_width, 8):
            context.move_to(x, self.y + self.padding)
            context.line_to(x, self.y + self.padding + self.inner_height)
        
        context.stroke()

        
        if not self.autoscale:

            for i in range(0, 110, 10): # 0,10,20..100
                
                value = i / 100.0
                y = self.y + self.padding + self.inner_height - self.inner_height * value

                context.move_to(self.x + self.padding, y)
                context.line_to(self.x + self.padding + self.inner_width, y)

            context.stroke()


        elif scale_factor > 0:
            
            if scale_factor > 1000:
                return # current maximum value under 1 permill, thus no guides are placed

            elif scale_factor > 100:
                # current maximum under 1 percent, place permill guides
                # TODO: set color from self/theme
                context.set_source_rgba(*self.colors['grid_milli'].tuple_rgba())
                for i in range(0, 10):
                    # place lines for 0-9 percent
                    value = i / 1000.0 * scale_factor
                    y = self.y + self.padding + self.inner_height - self.inner_height * value

                    if y < self.y + self.padding:
                        break # stop the loop if guides would be placed outside the gauge

                    context.move_to(self.x + self.padding, y)
                    context.line_to(self.x + self.padding + self.inner_width, y)
                
                context.stroke()

            elif scale_factor > 10:

                # TODO: set color from self/theme
                context.set_source_rgba(*self.colors['grid_minor'].tuple_rgba())
                for i in range(0, 10):
                    # place lines for 0-9 percent
                    value = i / 100.0 * scale_factor
                    y = self.y + self.padding + self.inner_height - self.inner_height * value

                    if y < self.y + self.padding:
                        break # stop the loop if guides would be placed outside the gauge

                    context.move_to(self.x + self.padding, y)
                    context.line_to(self.x + self.padding + self.inner_width, y)
                
                context.stroke()

            else: # major (10% step) guides
                # TODO: set color from self/theme
                context.set_source_rgba(*self.colors['grid_major'].tuple_rgba())
                for i in range(0, 110, 10): # 0,10,20..100
                    
                    value = i / 100.0 * scale_factor
                    y = self.y + self.padding + self.inner_height - self.inner_height * value

                    if y < self.y + self.padding:
                        break # stop the loop if guides would be placed outside the gauge

                    context.move_to(self.x + self.padding, y)
                    context.line_to(self.x + self.padding + self.inner_width, y)

                context.stroke()

        #context.set_dash([1,0]) # reset dash


    def update(self, context, monitor):
            
        self.points.append(monitor.normalized(self.address))
      
        context.set_line_width(1)
        context.set_source_rgba(*self.colors['background'].tuple_rgba())
        context.rectangle(self.x + self.padding, self.y + self.padding, self.inner_width, self.inner_height)
        context.fill()

        self.draw_grid(context, monitor)

        if self.autoscale:
            scale_factor = self.get_scale_factor()
            if scale_factor == 0.0:
                text = u"∞X"
            else:
                text = "%.2fX" % self.get_scale_factor()
            render_caption(context, text, self.x + self.padding + self.inner_width, self.y, align='right_top', color=(0.5,1,0,0.3), font_size=10)
        

        coords = []
        if self.autoscale:
            points = self.get_points_scaled()
        else:
            points = self.points

        for idx, amplitude in enumerate(points):
            coords.append((
                idx * 8 + self.padding,
                self.y + self.padding + self.inner_height - (self.inner_height * amplitude)
            ))
      
        context.set_source_rgba(*self.colors['foreground'].tuple_rgba())
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

            context.set_source_rgba(*self.colors['foreground'].tuple_rgba())

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

        context.set_source_rgba(*CONFIG_COLORS['window_background'].tuple_rgba())
        context.rectangle(0, 0, self.window.width, self.window.height)
        context.fill()

        for source, monitor in self.monitors.iteritems():

            if len(monitor.data) and self.gauges.has_key(source):
                gauges = self.gauges[source]

                for gauge in gauges:
                    gauge.update(context, monitor)


    def start(self):

        assert CONFIG_FPS != 0
        assert isinstance(CONFIG_FPS, float) or CONFIG_FPS >= 1

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
                'text': '{aggregate:.1f}% on {count} cores',
                'position': 'center_center',
                'align': 'center_center'
            }
        ]
    ), 

    ArcGauge(
        x=0,
        y=120,
        width=hugin.window.width / 2,
        height=100,
        address=0,
        captions=[
            {
                'text': '{core_0:.1f}%',
                'position': 'center_center',
                'align': 'center_center',
                'font_size': 10,
            }
        ]
    ),

    ArcGauge(
        x=hugin.window.width / 2,
        y=120,
        width=hugin.window.width / 2,
        height=100,
        address=1,
        captions=[
            {
                'text': '{core_1:.1f}%',
                'position': 'center_center',
                'align': 'center_center',
                'font_size': 10,
            }
        ]
    ),

    ArcGauge(
        x=0,
        y=210,
        width=hugin.window.width / 2,
        height=100,
        address=2,
        captions=[
            {
                'text': '{core_2:.1f}%',
                'position': 'center_center',
                'align': 'center_center',
                'font_size': 10,
            }
        ]
    ),

    ArcGauge(
        x=hugin.window.width / 2,
        y=210,
        width=hugin.window.width / 2,
        height=100,
        address=3,
        captions=[
            {
                'text': '{core_3:.1f}%',
                'position': 'center_center',
                'align': 'center_center',
                'font_size': 10,
            }
        ]
    ),

    PlotGauge(
        x=0,
        y=310,
        width=hugin.window.width,
        height=100,
        padding=15,
        autoscale=False,
    )
]

hugin.gauges['memory'] = [
    ArcGauge(
        x=0,
        y=410,
        width=hugin.window.width,
        height=hugin.window.width,
        stroke_width=30,
        captions=[
            {
                'text': '{percent:.1f}%',
                'position': 'center_center',
                'align': 'center_center'
            },

            {
                'text': '{total}',
                'position': 'left_top',
                'align': 'left_top',
                'font_size': 8,
            }
        ]
    )
]

hugin.gauges['network'] = [
    DualArcGauge(
        x=0,
        y=600,
        width=hugin.window.width,
        height=hugin.window.width,
        address=['em0.bytes_recv', 'em0.bytes_sent'],
        captions=[
            {
                'text': '{em0[counters][bytes_recv]}/s\n{em0[counters][bytes_sent]}/s',
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
        address='em0.bytes_recv',
    ),
    
    PlotGauge(
        x=0,
        y=900,
        width=hugin.window.width,
        height=100,
        padding=15,
        address='em0.bytes_sent'
    )
]

hugin.start()
