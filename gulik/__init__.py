#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import random
import signal
import collections
import queue
import functools
import threading
import multiprocessing
import setproctitle
import psutil
import colorsys
import cairo
import gi

gi.require_version('Gtk', '3.0')
gi.require_version('PangoCairo', '1.0') # not sure if want
from gi.repository import Gtk, Gdk, GLib, Pango, PangoCairo

from . import netdata

Operator = cairo.Operator

PAGESIZE = os.sysconf('SC_PAGESIZE')

## Helpers ##

class Color(object):

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


class DotDict(dict):

    """
    A dictionary with its data being readable through faked attributes.
    Used to avoid [[[][][][][]] in caption formatting.
    """

    def __getattribute__(self, name):


        #data = super(DotDict, self).__getattribute__('data')
        keys = super(DotDict, self).keys()
        if name in keys:
            return self.get(name)

        return super(DotDict, self).__getattribute__(name)


def palette_hue(base, count, distance=180):

    """
    Creates a hue-rotation palette 
    base - Color on which the palette will be based (i.e. the starting point of the hue-rotation)
    count - number of colors the palette should hold
    distance - angular distance on a 360° hue circle thingamabob
    """

    if count == 1:
        return [base]

    palette = []

    for i in range(0, count):
        color = base.clone()
        color.hue += i/(count - 1) * distance

        palette.append(color)

    return palette


def palette_value(base, count, min=None, max=None):
    
    """
    Creates a value-stepped palette 
    base - Color on which the palette will be based (i.e. source of hue and saturation)
    count - number of colors the palette should hold
    min - minimum value (the v in hsv)
    max - maxmimum value
    """

    if count == 1:
        return [base]

    if min is None:
        if 0.2 > base.value:
            min = base.value
        else:
            min = 0.2

    if max is None:
        if 0.6 < base.value:
            max = base.value
        else:
            max = 0.6

    span = max - min 
    step = span / (count - 1)

    palette = []

    for i in range(0, count):
        color = base.clone()
        color.value = max - i * step
        palette.append(color)

    return palette


def pretty_si(number):

    """
    Return a SI-postfixed string representation of a number (int or float).
    """

    postfixes = ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']

    value = number
    for postfix in postfixes:
        if value / 1000.0 < 1:
            break

        value /= 1000.0

    return "%.2f%s" % (value, postfix)


def pretty_bytes(bytecount):

    """
    Return a human-readable representation given a size in bytes.
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
    Return a human-readable representation in bits given a size in bytes.
    """

    units = ['bit', 'kbit', 'Mbit', 'Gbit', 'Tbit']

    value = bytecount * 8 # bytes to bits
    for unit in units:
        if value / 1024.0 < 1:
            break

        value /= 1024.0

    return "%.2f %s" % (value, unit)


def ignore_none(*args):

    """
    Return the first passed value that isn't None.
    """

    for arg in args:
        if not arg is None:
            return arg


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




## PATTERNS ##
def stripe45(color):

    surface = cairo.ImageSurface(cairo.Format.ARGB32, 10, 10)
    context = cairo.Context(surface)
    context.set_source_rgba(*color.tuple_rgba())
    context.move_to(5, 5)
    context.line_to(10, 0)
    context.line_to(10, 5)
    context.line_to(5, 10)
    context.line_to(0, 10)
    context.line_to(5, 5)
    context.close_path()
    context.fill()

    context.move_to(0, 0)
    context.line_to(5, 0)
    context.line_to(0, 5)
    context.close_path()
    context.fill()

    return surface

##class PeriodicCall(threading.Thread):
#
#    """ Periodically forces a window to redraw """
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


DEFAULTS = {
    'FPS': 1,
    'WIDTH': 200,
    'HEIGHT': Gdk.Screen().get_default().get_height(),
    'X': 0,
    'Y': 0,
    'NETDATA_HOSTS': [],
    'NETDATA_RETRY': 5,

    # styling stuff below this

    'MARGIN': 5,
    'PADDING': 5,
    'PADDING_BOTTOM': 40, # space for 2-rows of autolegend

    'FONT': 'Orbitron',
    'FONT_WEIGHT': 'Light',
    'FONT_SIZE': 10,

    'COLOR_WINDOW_BACKGROUND': Color(0,0,0, 0.6),
    'COLOR_BACKGROUND': Color(1,1,1, 0.1), # background for gauges
    'COLOR_FOREGROUND': Color(0.5, 1, 0, 0.6),
    'COLOR_CAPTION': Color(1,1,1, 0.6),
    #'COLOR_CAPTION_MINOR': Color(1,1,1, 0.3),
        
    'PALETTE': functools.partial(palette_hue, distance=-120), # mhh, curry…
    'PATTERN': stripe45,

    'CAPTION_PLACEMENT': 'inner', # allow captions to be properly centered in the inner region of gauges, as opposed to 'padding'
    #'CAPTION_PLACEMENT': 'padding', # allow captions to be placed within paddings, as opposed to 'inner'

    'LEGEND': True,
    'LEGEND_ORDER': 'normal', # other valid value: 'reverse'
    'LEGEND_SIZE': 20, # not font size, but height of one legend cell, including margin and padding.
    'LEGEND_PLACEMENT': 'padding',
    'LEGEND_MARGIN': 2.5,
    #'LEGEND_PADDING': 2.5,
    'LEGEND_PADDING': 0,

    'OPERATOR': Operator.OVER,

    # class-specific default styles
    
    'PADDING_TEXT': 0, # otherwise text will be tiny

    'PADDING_RECT': 5,
    'FONT_SIZE_RECT': 14,
    'FONT_WEIGHT_RECT': 'Bold',
    #'COLOR_RECT_CAPTION': Color(1,1,1, 1),
    'PATTERN_RECT': None,
    'OPERATOR_RECT_CAPTION': Operator.DIFFERENCE, # grants visibility no matter how much of the rect is filled
    'CAPTION_PLACEMENT_RECT': 'inner', # to get padding around cut-out

    'PATTERN_ARC': None,
    'PATTERN_MIRRORARC': None,
}

## Stuff I'd much rather do without a huge dependency like gtk ##
class Window(Gtk.Window):

    def __init__(self):

        super(Window, self).__init__()

        self.set_title('gulik')
        self.set_role('gulik')
        self.stick() # show this window on every virtual desktop

        self.set_app_paintable(True)
        self.set_type_hint(Gdk.WindowTypeHint.DOCK)
        self.set_keep_below(True)

        screen = self.get_screen()
        visual = screen.get_rgba_visual()
        if visual != None and screen.is_composited():
            self.set_visual(visual)

        self.show_all()


    @property
    def width(self):

        return self.get_size()[0]


    @property
    def height(self):

        return self.get_size()[1]


## Collectors ##

class Collector(multiprocessing.Process):

    def __init__(self, app, queue_update, queue_data):

        super(Collector, self).__init__()
        self.daemon = True
        self.app = app
        self.queue_update = queue_update
        self.queue_data = queue_data
        self.elements = []


    def terminate(self):

        #self.queue_data.close() # closing queues manually actually seems to mess stuff up
       
        # Would've done this cleaner, but after half a day of chasing some
        # retarded quantenbug I'm done with this shit. Just nuke the fucking
        # things from orbit.
        os.kill(self.pid, signal.SIGKILL)
        #super(Collector, self).terminate()


    def run(self):

        setproctitle.setproctitle(f"gulik - {self.__class__.__name__}")

        while True:

            try:
                msg = self.queue_update.get(block=True)
                if msg == 'UPDATE':
                    self.update()

            except KeyboardInterrupt: # so we don't randomly explode on ctrl+c
                pass


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
            },
            block=True
        )
        
        # according to psutil docs, there should at least be 0.1 seconds
        # between calls to cpu_percent without sampling interval
        time.sleep(0.1) 


class MemoryCollector(Collector):

    def update(self):

        vmem = psutil.virtual_memory()

        processes = []
        total_use = 0
        for process in psutil.process_iter():

            if psutil.LINUX:
                pmem = process.memory_full_info()

                processes.append(DotDict({
                    'name': process.name(),
                    'private': pmem.pss,
                    'shared': pmem.shared,
                    'percent': pmem.pss / vmem.total * 100
                }))

                total_use += pmem.pss

            elif psutil.BSD:

                try:
                    resident = 0
                    private = 0
                    #shared = 0
                        
                    try:
                        for mmap in process.memory_maps():

                            # assuming everything with a real path is
                            # "not really in ram", but no clue.
                            if  mmap.path.startswith('['): 
                                private += mmap.private * PAGESIZE
                                resident += mmap.rss * PAGESIZE
                                #shared += (mmap.rss - mmap.private) * PAGESIZE # FIXME: probably broken, can yield negative values
                    except OSError:
                        pass # probably "device not available"

                    processes.append(DotDict({
                        'name': process.name(),
                        'private': private,
                        'shared': resident - private,
                        'percent': private / vmem.total * 100,
                    }))
                    
                    total_use += private

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                    pass# TODO: add counter for processes we can't introspect

        info = DotDict({
            'total': vmem.total,
            'percent': total_use / vmem.total * 100,
            'available': vmem.total - total_use
        })

        processes_sorted = sorted(processes, key=lambda x: x['private'], reverse=True)

        for i, process in enumerate(processes_sorted[:3]):
            info['top_%d' % (i + 1)] = process 

        info['other'] = DotDict({
            'name': 'other',
            'private': 0,
            'shared': 0,
            'count': 0
        })

        for process in processes_sorted[3:]:

            info['other']['private'] += process['private']
            info['other']['shared'] += process['shared']
            info['other']['count'] += 1

        info['other']['percent'] = info['other']['private'] / vmem.total * 100

        self.queue_data.put(info, block=True)

        #time.sleep(1) # because this became horribly slow


class NetworkCollector(Collector):

    def update(self):

        stats = psutil.net_if_stats()
        addrs = psutil.net_if_addrs()
        counters = psutil.net_io_counters(pernic=True)
        connections = psutil.net_connections(kind='all')

        self.queue_data.put(
            {
                'stats': stats,
                'addrs': addrs,
                'counters': counters,
                'connections': connections,
            },
            block=True
        )


class BatteryCollector(Collector):

    def update(self):
        
        self.queue_data.put(psutil.sensors_battery(), block=True)


class NetdataCollector(Collector):

    def __init__(self, app, queue_update, queue_data, host, port):

        super(NetdataCollector, self).__init__(app, queue_update, queue_data)
        self.client = netdata.Netdata(host, port=port, timeout=1/self.app.config['FPS'])


    def run(self):

        setproctitle.setproctitle(f"gulik - {self.__class__.__name__}")

        while True:

            try:
                msg = self.queue_update.get(block=True)
                if msg.startswith('UPDATE '):
                    chart = msg[7:]
                    self.update(chart)

            except KeyboardInterrupt: # so we don't randomly explode on ctrl+c
                pass


    def update(self, chart):

        try:
            # get the last second of data condensed to one point
            data = self.client.data(chart, points=1, after=-1, options=['absolute'])
        except netdata.NetdataException:
            pass
        else:
            self.queue_data.put((chart, data), block=True)


## Monitors ##

class Monitor(threading.Thread):

    collector_type = Collector

    def __init__(self, app, component):

        super(Monitor, self).__init__()
        self.app = app
        self.component = component
        self.daemon = True
        self.seppuku = False

        self.queue_update = multiprocessing.Queue(1)
        self.queue_data = multiprocessing.Queue(1)
        self.collector = self.collector_type(self.app, self.queue_update, self.queue_data)
        self.data = {}
        self.defective = False # for future use, mostly for networked monitors (netdata, mpd, …)


    def register_elements(self, elements):
        pass


    def tick(self):
            
        if not self.queue_update.full():
            self.queue_update.put('UPDATE', block=True)


    def start(self):

        self.collector.start()
        super(Monitor, self).start()


    def run(self):

        #while self.collector.is_alive():
        while not self.seppuku:

            try:
                self.data = self.queue_data.get(timeout=1)
            except queue.Empty:
                # try again, but give thread the ability to die without
                # waiting on collector indefinitely
                continue 
        self.commit_seppuku()


    def commit_seppuku(self):

        print(f"{self.__class__.__name__} committing glorious seppuku!")

        #self.queue_update.close()
        self.collector.terminate()
        self.collector.join()


    def normalize(self, element):
        raise NotImplementedError("%s.normalize not implemented!" % self.__class__.__name__)


    def caption(self, fmt):
        raise NotImplementedError("%s.caption not implemented!" % self.__class__.__name__)


class CPUMonitor(Monitor):

    collector_type = CPUCollector

    def normalize(self, element):

        if not self.data:
            return 0

        if element == 'aggregate':
            return self.data['aggregate'] / 100.0
        
        # assume core_<n> otherwise
        idx = int(element.split('_')[1])
        return self.data['percpu'][idx] / 100.0


    def caption(self, fmt):

        if not self.data:
            return fmt

        data = {}
        data['count'] = self.data['count']
        data['aggregate'] = self.data['aggregate']
        for idx, perc in enumerate(self.data['percpu']):
            data['core_%d' % idx] = perc

        return fmt.format(**data)


class MemoryMonitor(Monitor):

    collector_type = MemoryCollector

    def normalize(self, element):

        if not self.data:
            return 0

        if element == 'percent':

            return self.data.get('percent', 0) / 100.0

        return self.data[element].get('percent', 0) / 100.0


    def caption(self, fmt):

        if not self.data:
            return fmt

        data = DotDict()#dict(self.data) # clone

        data['total'] = pretty_bytes(self.data['total'])
        data['available'] = pretty_bytes(self.data['available'])
        data['percent'] = self.data['percent']
        
        for k in ['top_1', 'top_2', 'top_3', 'other']:

            data[k] = DotDict()
            data[k]['name'] = self.data[k]['name']
            data[k]['private'] = pretty_bytes(self.data[k]['private'])
            data[k]['shared'] = pretty_bytes(self.data[k]['shared'])
            if k == 'other':
                data[k]['count'] = self.data[k]['count']

        return fmt.format(**data)


class NetworkMonitor(Monitor):

    collector_type = NetworkCollector

    def __init__(self, app, component):

        super(NetworkMonitor, self).__init__(app, component)

        self.interfaces = collections.OrderedDict()

        if self.app.config['FPS'] < 2:
            # we need a minimum of 2 samples so we can compute a difference
            deque_len = 2 
        else:
            # max size equal fps means this holds data of only the last second
            deque_len = self.app.config['FPS']

        keys = [
            'bytes_sent',
            'bytes_recv',
            'packets_sent',
            'packets_recv',
            'errin',
            'errout',
            'dropin',
            'dropout'
        ]

        for if_name in psutil.net_if_stats().keys():

            self.interfaces[if_name] = {
                'addrs': {},
                'stats': {},
                'counters': {}
            }
            for key in keys:
                self.interfaces[if_name]['counters'][key] = collections.deque([], deque_len)

        self.aggregate = {
            'if_count': len(self.interfaces),
            'if_up': 0,
            'speed': 0, # aggregate link speed
            'counters': {}
        }

        for key in keys:
            self.aggregate['counters'][key] = collections.deque([], deque_len)


    def run(self):

        while not self.seppuku:

            try:
                self.data = self.queue_data.get(timeout=1)
            except queue.Empty:
                # try again, but give thread the ability to die
                # without waiting on collector indefinitely.
                continue             

            aggregates = {}
            for key in self.aggregate['counters']:
                #self.aggregate['counters'][k] = []
                aggregates[key] = 0

            self.aggregate['speed'] = 0
            for if_name, if_data in self.interfaces.items():

                if_has_data = if_name in self.data['counters'] and\
                    if_name in self.data['stats'] and\
                    if_name in self.data['addrs']

                if if_has_data:

                    for key, deque in if_data['counters'].items():
                        value = self.data['counters'][if_name]._asdict()[key]
                        deque.append(value)
                        aggregates[key] += value
                    self.interfaces[if_name]['stats'] = self.data['stats'][if_name]._asdict()
                    if self.interfaces[if_name]['stats']['speed'] == 0:
                        self.interfaces[if_name]['stats']['speed'] = 1000 # assume gbit speed per default

                    self.aggregate['speed'] += self.interfaces[if_name]['stats']['speed']
                    
                    if if_name in self.data['addrs']:
                        self.interfaces[if_name]['addrs'] = self.data['addrs'][if_name]
                    else:
                        self.interfaces[if_name]['addrs'] = []

            for key, value in aggregates.items():
                self.aggregate['counters'][key].append(value)

        self.commit_seppuku()


    def count_sec(self, interface, key):

        """
            get a specified count for a given interface
            as calculated for the last second.

            EXAMPLE: self.count_sec('eth0', 'bytes_sent') 
            (will return count of bytes sent in the last second)
        """

        if interface == 'aggregate':
            deque = self.aggregate['counters'][key]
        else:
            deque = self.interfaces[interface]['counters'][key]
        
        if self.app.config['FPS'] < 2:
            # fps < 1 means data covers 1/fps seconds
            return (deque[-1] - deque[0]) / self.app.config['FPS']
        else:
            # last (most recent) minus first (oldest) item
            return deque[-1] - deque[0]


    def normalize(self, element):

        if_name, key = element.split('.')

        if if_name == 'aggregate':
            if len(self.aggregate['counters'][key]) >= 2:
                link_quality = float(self.aggregate['speed'] * 1024**2)
                return (self.count_sec(if_name, key) * 8) / link_quality

        elif len(self.interfaces[if_name]['counters'][key]) >= 2:
            link_quality = float(self.interfaces[if_name]['stats']['speed'] * 1024**2)

            return (self.count_sec(if_name, key) * 8) / link_quality

        # program flow should only arrive here if we have less than 2
        # datapoints in which case we can't establish used bandwidth.
        return 0 


    def caption(self, fmt):

        if not self.data:
            return fmt
        
        data = {}

        data['aggregate'] = DotDict()
        data['aggregate']['if_count'] = self.aggregate['if_count']
        data['aggregate']['if_up'] = self.aggregate['if_up']
        data['aggregate']['speed'] = self.aggregate['speed']
        data['aggregate']['counters'] = DotDict()

        for key in self.aggregate['counters'].keys():

            data['aggregate']['counters'][key] = self.count_sec('aggregate', key)
            if key.startswith('bytes'):
                data['aggregate']['counters'][key] = pretty_bits(data['aggregate']['counters'][key]) + '/s'

        for if_name in self.interfaces.keys():

            data[if_name] = DotDict()
            data[if_name]['addrs'] = DotDict()
            all_addrs = []
            for idx, addr in enumerate(self.interfaces[if_name]['addrs']):
                data[if_name]['addrs'][str(idx)] = addr
                all_addrs.append(addr.address)

            data[if_name]['all_addrs'] = u"\n".join(all_addrs)

            data[if_name]['stats'] = DotDict(self.interfaces[if_name]['stats'])

            data[if_name]['counters'] = DotDict()
            for key in self.interfaces[if_name]['counters'].keys():

                data[if_name]['counters'][key] = self.count_sec(if_name, key)
                if key.startswith('bytes'):
                    data[if_name]['counters'][key] = pretty_bits(data[if_name]['counters'][key]) + '/s'

        return fmt.format(**data)


class BatteryMonitor(Monitor):

    collector_type = BatteryCollector

    def normalize(self, element):

        # TODO: multi-battery support? needs support by psutil…

        if not self.data:
            return 0

        return self.data.percent / 100.0


    def caption(self, fmt):

        if not self.data:
            return fmt

        data = self.data._asdict()

        if not data['power_plugged']:
            data['state'] = 'draining'
        elif data['percent'] == 100:
            data['state'] = 'Ffll'
        else:
            data['state'] = 'charging'

        return fmt.format(**data)


class NetdataMonitor(Monitor):

    collector_type = NetdataCollector

    def __init__(self, app, component, host, port):

        self.collector_type = functools.partial(self.collector_type, host=host, port=port)

        super(NetdataMonitor, self).__init__(app, component)

        self.charts = set()
        self.normalization_values = {} # keep a table of known maximums because netdata doesn't supply absolute normalization values
       
        self.info_last_try = time.time()
        try:
            self.netdata_info = self.collector.client.charts()
        except netdata.NetdataException as e:
            self.netdata_info = None
            self.defective = True


    def __repr__(self):

        return f"<{self.__class__.__name__} host={self.collector.client.host} port={self.collector.client.port}>"


    def register_elements(self, elements):

        for element in elements:
            parts = element.split('.')
            chart = '.'.join(parts[:2])

            if not chart in self.charts:

                self.normalization_values[chart] = 0

                if self.netdata_info:
                    if not chart in self.netdata_info['charts']:
                        raise ValueError(f"Invalid chart: {chart} on netdata instance {self.host}:{self.port}!")

                    chart_info = self.netdata_info['charts'][chart]
                    if chart_info['units'] == 'percentage':
                        self.normalization_values[chart] = 100
                    else:
                        self.normalization_values[chart] = 0

                self.charts.add(chart)

    
    def run(self):

        #while self.collector.is_alive():
        while not self.seppuku:

            try:
                (chart, data) = self.queue_data.get(timeout=1/self.app.config['FPS'])
                self.data[chart] = data

                if self.netdata_info['charts'][chart]['units'] != 'percentage':

                    cumulative_value = sum(data['data'][0][1:])
                    if self.normalization_values[chart] < cumulative_value:
                        self.normalization_values[chart] = cumulative_value

            except queue.Empty:
                continue # try again

        self.commit_seppuku()


    def tick(self):

        if self.defective:

            t = time.time()
            if t >= self.info_last_try + self.app.config['NETDATA_RETRY']:
                print(f"{self.__class__.__name__} instance currently defective, trying to get netdata overview from {self.collector.client.host}.")
                self.info_last_try = t
                try:
                    self.netdata_info = self.collector.client.charts()
                    self.defective = False
                    self.tick() # do the actual tick (i.e. the else clause)
                except netdata.NetdataException as e:
                    print(f"Failed, will retry in {self.app.config['NETDATA_RETRY']} seconds.")

        else:
            if not self.queue_update.full():
            #if not self.seppuku: # don't request more updates to collector when we're trying to die
                for chart in self.charts:
                    self.queue_update.put(f"UPDATE {chart}", block=True)


    def normalize(self, element):

        parts = element.split('.')

        chart = '.'.join(parts[:2])

        #if chart not in self.charts or not self.data[chart]:
        if not chart in self.data:
            #print(f"No data for {chart}")
            return 0 #

        #timestamp = self.data[chart]['data'][0][0] # first element of a netdata datapoint is always time
        #if timestamp > self.last_updates[chart]:

        subelem = parts[2]
        subidx = self.data[chart]['labels'].index(subelem)
        value = self.data[chart]['data'][0][subidx]

        if value >= self.normalization_values[chart]:
            self.normalization_values[chart] = value

        if self.normalization_values[chart] == 0:
            return 0
        r = value / self.normalization_values[chart]
        return r


    def caption(self, fmt):

        if not self.data:
            return fmt

        data = DotDict()

        for chart_name, chart_data in self.data.items():

            chart_keys = chart_name.split('.')
            unit = self.netdata_info['charts'][chart_name]['units'] # called "units" but actually only ever one. it's a string.

            if not chart_keys[0] in data:
                data[chart_keys[0]] = DotDict()

            d = DotDict()

            for idx, label in enumerate(chart_data['labels']):
                value = chart_data['data'][0][idx]

                if unit == 'bytes':
                    value = pretty_bytes(value)

                elif unit.startswith('kilobytes'):

                    postfix = unit[9:]
                    value = pretty_bytes(value * 1024) + postfix

                elif unit.startswith('kilobits'):
                    postfix = unit[8:]
                    value = pretty_bits(value * 1024) + postfix

                else:
                    value = f"{value} {unit}"

                d[label] = value
            
            data[chart_keys[0]][chart_keys[1]] = d

        return fmt.format(**data)


## Gauges ##

class Gauge(object):

    def __init__(
        self,
        app,
        monitor,
        x=0,
        y=0,
        width=None,
        height=None,
        margin=None,
        margin_left=None,
        margin_right=None,
        margin_top=None,
        margin_bottom=None,
        padding=None,
        padding_left=None,
        padding_right=None,
        padding_top=None,
        padding_bottom=None,
        elements=None,
        captions=None,
        caption_placement=None,
        legend=None,
        legend_order=None,
        legend_format=None,
        legend_size=None,
        legend_placement=None,
        legend_margin=None,
        legend_margin_left=None,
        legend_margin_right=None,
        legend_margin_top=None,
        legend_margin_bottom=None,
        legend_padding=None,
        legend_padding_left=None,
        legend_padding_right=None,
        legend_padding_top=None,
        legend_padding_bottom=None,
        foreground=None,
        background=None,
        pattern=None,
        palette=None,
        combination=None,
        operator=None
    ):

        self.app = app
        self.monitor = monitor
        self.x = x
        self.y = y
        self.elements = ignore_none(elements, [])
        self.captions = ignore_none(captions, list())
        self.caption_placement = ignore_none(caption_placement, self.get_style('caption_placement'))

        self.legend = ignore_none(legend, self.get_style('legend'))
        self.legend_order = ignore_none(legend_order, self.get_style('legend_order'))
        self.legend_format = legend_format
        self.legend_placement = ignore_none(legend_placement, self.get_style('legend_placement'))
        self.legend_size= ignore_none(legend_size, self.get_style('legend_size'))
        self.legend_margin_left = ignore_none(legend_margin_left, legend_margin, self.get_style('legend_margin', 'left'))
        self.legend_margin_right = ignore_none(legend_margin_right, legend_margin, self.get_style('legend_margin', 'right'))
        self.legend_margin_top = ignore_none(legend_margin_top, legend_margin, self.get_style('legend_margin', 'top'))
        self.legend_margin_bottom = ignore_none(legend_margin_bottom, legend_margin, self.get_style('legend_margin', 'bottom'))
        self.legend_padding_left = ignore_none(legend_padding_left, legend_padding, self.get_style('legend_padding', 'left'))
        self.legend_padding_right = ignore_none(legend_padding_right, legend_padding, self.get_style('legend_padding', 'right'))
        self.legend_padding_top = ignore_none(legend_padding_top, legend_padding, self.get_style('legend_padding', 'top'))
        self.legend_padding_bottom = ignore_none(legend_padding_bottom, legend_padding, self.get_style('legend_padding', 'bottom'))
        
        self.operator = ignore_none(operator, self.get_style('operator'))
        
        self.width = ignore_none(width, self.get_style('width'))
        self.height = ignore_none(height, self.get_style('height'))
        
        self.margin_left = ignore_none(margin_left, margin, self.get_style('margin', 'left'))
        self.margin_right = ignore_none(margin_right, margin, self.get_style('margin', 'right'))
        self.margin_top = ignore_none(margin_top, margin, self.get_style('margin', 'top'))
        self.margin_bottom = ignore_none(margin_bottom, margin, self.get_style('margin', 'bottom'))

        self.padding_left = ignore_none(padding_left, padding, self.get_style('padding', 'left'))
        self.padding_right = ignore_none(padding_right, padding, self.get_style('padding', 'right'))
        self.padding_top = ignore_none(padding_top, padding, self.get_style('padding', 'top'))
        self.padding_bottom = ignore_none(padding_bottom, padding, self.get_style('padding', 'bottom'))
        
        self.colors = {}
        self.colors['foreground'] = ignore_none(foreground, self.get_style('color', 'foreground'))
        self.colors['background'] = ignore_none(background, self.get_style('color', 'background'))

        self.pattern = ignore_none(pattern, self.get_style('pattern'))
        self.palette = ignore_none(palette, self.get_style('palette')) # function to generate color palettes with

        self.combination = ignore_none(combination, 'separate') # combination mode when handling multiple elements. 'separate', 'cumulative' or 'cumulative_force'. cumulative assumes all values add up to max 1.0, while separate assumes every value can reach 1.0 and divides all values by the number of elements handled

        assert self.inner_width > 0, f"margin and padding too big, implying negative inner width for {self.__class__.__name__}/{self.monitor.component}/{self.elements}"

        assert self.inner_height > 0, f"margin and padding too big, implying negative inner height for {self.__class__.__name__}/{self.monitor.component}/{self.elements}"


        self.monitor.register_elements(self.elements)

        if self.legend:

            legend_x = self.x + self.margin_left + self.padding_left
            legend_y = self.y + self.margin_top + self.padding_top
            if self.legend_placement == 'inner':
                legend_height = self.inner_height
            else: # 'padding'
                legend_y += self.inner_height
                legend_height = self.padding_bottom

            rownum = legend_height // self.legend_size

            if rownum < 1:
                print(f"Can't add autolegend to {self.__class__.__name__}/{self.monitor.component} because of insufficient space: {legend_height}px")
            else:

                colnum = math.ceil(len(self.elements) // rownum) or 1
                cell_width = self.inner_width // colnum

                #colors = self.palette(self.colors['foreground'], len(self.elements))

                box = self.app.box(legend_x, legend_y, self.inner_width, legend_height)

                legend_info = self.legend_info()

                if self.legend_order == 'reverse':
                    iterator = reversed(legend_info)
                else:
                    iterator = legend_info

                #for element in legend_elements:
                for element in iterator:

                    color, text = legend_info[element]
                    if self.legend_format:
                        text = self.legend_format.format(**{'element': element})

                    box.place(
                        self.monitor.component,
                        Text,
                        text=text,
                        align='center_center',
                        foreground=color,
                        background=Color(0,0,0, 0),
                        width=cell_width,
                        height=self.legend_size,
                        margin_left=self.legend_margin_left,
                        margin_right=self.legend_margin_right,
                        margin_top=self.legend_margin_top,
                        margin_bottom=self.legend_margin_bottom,
                        padding_left=self.legend_padding_left,
                        padding_right=self.legend_padding_right,
                        padding_top=self.legend_padding_top,
                        padding_bottom=self.legend_padding_bottom,
                        legend=False
                    )


    def get_style(self, name, subname=None):

        """
        load the most specific style setting available given a name and optional subname.
        usage examples: self.get_style('margin', 'left'), 
        """

        keys = []
        if subname:
            keys.append('_'.join([name, self.__class__.__name__, subname]).upper())
        keys.append('_'.join([name, self.__class__.__name__]).upper())
        if subname:
            keys.append('_'.join([name, subname]).upper())
        keys.append(name.upper())

        for key in keys:
            if key in self.app.config:
                return self.app.config[key]


    def legend_info(self):

        """ defines colors for legend elements """

        data = collections.OrderedDict()
        
        colors = self.palette(self.colors['foreground'], len(self.elements))
        for idx, color in enumerate(colors):
            element = self.elements[idx]
            data[element] = (color, element)

        return data



    @property
    def padded_width(self):
        return self.width - self.margin_left - self.margin_right


    @property
    def padded_height(self):
        return self.height - self.margin_top - self.margin_bottom


    @property
    def inner_width(self):
        return self.padded_width - self.padding_left - self.padding_right


    @property
    def inner_height(self):
        return self.padded_height - self.padding_top - self.padding_bottom


    def set_brush(self, context, color): # possible TODO: better function name

        if self.pattern:
            context.set_source_surface(self.pattern(color))
            context.get_source().set_extend(cairo.Extend.REPEAT)

        else:
            context.set_source_rgba(*color.tuple_rgba())


    def draw_background(self, context):

        context.set_source_rgba(*self.colors['background'].tuple_rgba())
        context.rectangle(self.x + self.margin_left+ self.padding_left, self.y + self.margin_top + self.padding_top, self.inner_width, self.inner_height)
        context.fill()


    def draw_captions(self, context):

        for caption in self.captions:

            context.save()
            context.set_operator(caption.get('operator', self.get_style('operator', 'caption')))

            if 'position' in caption:

                if isinstance(caption['position'], str):
                    # handle alignment-style strings like "center_bottom"

                    if self.caption_placement == 'inner':
                        offset = [-x for x in alignment_offset(caption['position'], (self.inner_width, self.inner_height))]
                    else:
                        offset = [-x for x in alignment_offset(caption['position'], (self.padded_width, self.padded_height))]

                else:
                    offset = caption['position']

            else:
                offset = [0, 0]

            if self.caption_placement == 'inner':
                position = [offset[0] + self.x + self.margin_left + self.padding_left, offset[1] + self.y + self.margin_top + self.padding_top]
            else:
                position = [offset[0] + self.x + self.margin_left, offset[1] + self.y + self.margin_top]

            caption_text = self.monitor.caption(caption['text'])

            self.app.draw_text(
                context,
                caption_text,
                position[0],
                position[1],
                align=caption.get('align', None),
                color=caption.get('color', self.get_style('color', 'caption')),
                font_size=caption.get('font_size', self.get_style('font_size', 'caption')),
                font_weight=caption.get('font_weight', self.get_style('font_weight', 'caption'))
            )

            context.restore()


    def update(self, context):

        """
        parameters:
            context: cairo context of the window
        """

        context.save()
        context.set_operator(self.operator)
        self.draw(context)
        context.restore()


    def draw(self, context):

        raise NotImplementedError("%s.draw not implemented!" % self.__class__.__name__)


class Text(Gauge):

    def __init__(self, app, monitor, text, speed=25, align=None, **kwargs):
       
        super(Text, self).__init__(app, monitor, **kwargs)
        self.text = text # the text to be rendered, a format string passed to monitor.caption
        self.previous_text = '' # to be able to detect change

        self.speed = speed

        self.align = ignore_none(align, 'left_top')

        surface = cairo.ImageSurface(cairo.Format.ARGB32, 10, 10)
        context = cairo.Context(surface)
        font = Pango.FontDescription('%s %s 10' % (self.get_style('font'), self.get_style('font_weight')))
        layout = PangoCairo.create_layout(context)
        layout.set_font_description(font)
        layout.set_text('0', -1) # naively assuming 0 is the highest glyph
        size = layout.get_pixel_size()
        if size[1] > 10:
            self.font_size = self.inner_height * 10/size[1]
        else:
            self.font_size = self.inner_height # probably not gonna happen

        self.direction = 'left'
        self.offset = 0.0
        self.step = speed / self.app.config['FPS'] # i.e. speed in pixel/s


    def draw_background(self, context):

        context.set_source_rgba(*self.colors['background'].tuple_rgba())
        context.rectangle(self.x + self.margin_left, self.y + self.margin_top, self.padded_width, self.padded_height)
        context.fill()


    def draw(self, context):
        
        text = self.monitor.caption(self.text)

        self.draw_background(context)

        context.save()
        context.rectangle(self.x + self.margin_left + self.padding_left, self.y + self.margin_top + self.padding_top, self.inner_width, self.inner_height)
        context.clip()

        context.set_source_rgba(*self.colors['foreground'].tuple_rgba())
        font = Pango.FontDescription('%s %s %d' % (self.get_style('font'), self.get_style('font_weight'), self.font_size))

        layout = PangoCairo.create_layout(context)
        layout.set_font_description(font)
        layout.set_text(text, -1)
        
        size = layout.get_pixel_size()
        align_offset = alignment_offset(self.align, size) # needed for the specified text alignment
        #align_offset = [sum(x) for x in zip(alignment_offset(self.align, size), alignment_offset(self.align, [self.inner_width, self.inner_height]))] # needed for the specified text alignment

        max_offset = size[0] - self.inner_width # biggest needed offset for marquee, can be negative if all text fits

        if max_offset <= 0 or text != self.previous_text:
            self.direction = 'left'
            self.offset = 0

        x = self.x + self.margin_left + self.padding_left - self.offset
       
        if max_offset < 0: # only account for horizontal offset when space allows

            x += align_offset[0]
            if self.align.startswith('center'):
                x += self.inner_width / 2
            
            elif self.align.startswith('right'):
                x += self.inner_width


        y = self.y + self.margin_top + self.padding_top + align_offset[1] # NOTE: does stuff even show up with vertical align != center?

        if self.align.endswith('center'):
            y += self.inner_height / 2

        elif self.align.endswith('bottom'):
            y += self.inner_height

        context.translate(x, y)

        PangoCairo.update_layout(context, layout)
        PangoCairo.show_layout(context, layout)

        context.restore()

        if self.direction == 'left':
            self.offset += self.step
            if self.offset > max_offset:
                self.direction = 'right'
                self.offset = max_offset

        else:
            self.offset -= self.step
            if self.offset < 0:
                self.direction = 'left'
                self.offset = 0

        self.previous_text = text


class Rect(Gauge):

    def draw_rect(self, context, value, color, offset=0.0):

            self.set_brush(context, color)

            context.rectangle(
                self.x + self.margin_left + self.padded_width * offset,
                self.y + self.margin_top,
                self.padded_width * value,
                self.padded_height)

            context.fill()


    def draw_background(self, context):

        self.draw_rect(context, 1, self.colors['background'])


    def draw(self, context):
        
        colors = self.palette(self.colors['foreground'], len(self.elements))
        offset = 0.0

        self.draw_background(context)

        for idx, element in enumerate(self.elements):

            value = self.monitor.normalize(element)

            if self.combination != 'cumulative':
                value /= len(self.elements)

            color = colors[idx]

            self.draw_rect(context, value, color, offset)

            if self.combination.startswith('cumulative'):
                offset += value
            else:
                offset += 1.0 / len(self.elements)

        self.draw_captions(context)


class MirrorRect(Gauge):

    def __init__(self, app, monitor, **kwargs):

        self.left = kwargs['elements'][0]
        self.right = kwargs['elements'][1]
        kwargs['elements'] = self.left + self.right
        super(MirrorRect, self).__init__(app, monitor, **kwargs)
        self.x_center = self.x + self.margin_left + (self.inner_width / 2)
        self.draw_left = self.draw_rect_negative
        self.draw_right = self.draw_rect
    
    
    def legend_info(self):

        """ defines colors for legend elements """

        data = {}
      
        colors = self.palette(self.colors['foreground'], max(len(self.left), len(self.right)))

        for elements in (self.left, self.right):
            for idx, element in enumerate(elements):
                data[element] = (colors[idx], element)

        return data

    
    def draw_rect(self, context, value, color, offset=0.0):

            self.set_brush(context, color)
            
            context.rectangle(self.x_center + self.inner_width / 2 * offset, self.y + self.margin_top + self.padding_top, self.inner_width / 2 * value, self.inner_height)
            context.fill()


    def draw_rect_negative(self, context, value, color, offset=0.0):

            self.set_brush(context, color)
            
            context.rectangle(self.x_center - self.inner_width / 2 * offset - self.inner_width / 2 * value, self.y + self.margin_top + self.padding_top, self.inner_width / 2 * value, self.inner_height)
            context.fill()


    def draw(self, context):

        colors = self.palette(self.colors['foreground'], max(len(self.left), len(self.right)))

        self.draw_background(context)

        for elements, drawer in ((self.left, self.draw_left), (self.right, self.draw_right)):

            offset = 0.0

            for idx, element in enumerate(elements):
                
                value = self.monitor.normalize(element)

                if self.combination != 'cumulative':
                    value /= len(elements)

                color = colors[idx]

                drawer(context, value, color, offset)

                if self.combination.startswith('cumulative'):
                    offset += value
                else:
                    offset += 1.0 / len(elements)

        self.draw_captions(context)


class Arc(Gauge):

    def __init__(self, app, monitor, stroke_width=5, **kwargs):

        super(Arc, self).__init__(app, monitor, **kwargs)
        self.stroke_width = stroke_width
        #self.radius = (min(self.width, self.height) / 2) - (2 * self.padding) - (self.stroke_width / 2)
        self.radius = (min(self.inner_width, self.inner_height) / 2) - self.stroke_width 
        self.x_center = self.x + self.margin_left + self.padding_left + (self.inner_width / 2)
        self.y_center = self.y + self.margin_top + self.padding_top + (self.inner_height / 2)


    def draw_arc(self, context, value, color, offset=0.0):

        context.set_line_width(self.stroke_width)
        context.set_line_cap(cairo.LINE_CAP_BUTT)

        self.set_brush(context, color)

        context.arc(
            self.x_center,
            self.y_center,
            self.radius,
            math.pi / 2 + math.pi * 2 * offset,
            math.pi / 2 + math.pi * 2 * (offset + value)
        )

        context.stroke()

    
    def draw_background(self, context):

        self.draw_arc(context, 1, self.colors['background'])


    def draw(self, context):

        self.draw_background(context)

        colors = self.palette(self.colors['foreground'], len(self.elements))
        offset = 0.0
        for idx, element in enumerate(self.elements):

            value = self.monitor.normalize(element)

            if self.combination != 'cumulative':
                value /= len(self.elements)

            color = colors[idx]

            self.draw_arc(context, value, color, offset=offset)

            if self.combination == 'separate':
                offset += 1 / len(self.elements)
            else:
                offset += value

        self.draw_captions(context)


class MirrorArc(MirrorRect, Arc):

    def __init__(self, app, monitor, **kwargs):

        super(MirrorArc, self).__init__(app, monitor, **kwargs)
        self.draw_left = self.draw_arc_negative
        self.draw_right = self.draw_arc


    def draw_arc(self, context, value, color, offset=0.0):

        value /= 2
        offset /= 2

        super(MirrorArc, self).draw_arc(context, value, color, offset=offset)


    def draw_arc_negative(self, context, value, color, offset=0.0):

        context.set_line_width(self.stroke_width)
        context.set_line_cap(cairo.LINE_CAP_BUTT)
        self.set_brush(context, color)

        context.arc_negative(
            self.x_center,
            self.y_center,
            self.radius,
            math.pi / 2 - math.pi * offset,
            math.pi / 2 - math.pi * (offset + value)
        )

        context.stroke()


    def draw_background(self, context):

        self.draw_arc(context, 2, self.colors['background'])


class Plot(Gauge):

    def __init__(self, app, monitor, num_points=None, autoscale=True, markers=True, line=True, grid=True, **kwargs):

        super(Plot, self).__init__(app, monitor, **kwargs)

        if num_points:
            self.num_points = num_points
            self.step = self.inner_width / (num_points - 1)
            assert int(self.step) >= 1, "num_points %d exceeds pixel density!" % num_points

        else:
            self.step = 8
            self.num_points = int(self.inner_width // self.step + 1)

        self.prepare_points() # creates self.points with a deque for every plot

        self.autoscale = autoscale
        self.markers = markers
        self.line = line
        self.grid = grid
        self.grid_height = self.inner_height

        self.colors['plot_line'] = self.colors['foreground'].clone()
        self.colors['plot_line'].alpha *= 0.8

        self.colors['plot_fill'] = self.colors['foreground'].clone()
        if line:
            self.colors['plot_fill'].alpha *= 0.25

        self.colors['grid_major'] = self.colors['plot_line'].clone()
        self.colors['grid_major'].alpha *= 0.5

        self.colors['grid_minor'] = self.colors['background'].clone()
        self.colors['grid_minor'].alpha *= 0.8

        self.colors['grid_milli'] = self.colors['background'].clone()
        self.colors['grid_milli'].alpha *= 0.4

        self.colors['caption_scale'] = self.colors['foreground'].clone()
        self.colors['caption_scale'].alpha *= 0.6

        self.colors_plot_marker = self.palette(self.colors['foreground'], len(self.elements))
        self.colors_plot_line = self.palette(self.colors['plot_line'], len(self.elements))
        self.colors_plot_fill = self.palette(self.colors['plot_fill'], len(self.elements))


    def prepare_points(self):

        self.points = collections.OrderedDict()
        for element in self.elements:
            self.points[element] = collections.deque([], self.num_points)


    def get_scale_factor(self, elements=None):

        if elements is None:
            elements = self.elements

        if self.combination.startswith('cumulative'):

            cumulative_points = []
            for idx in range(0, self.num_points):
                value = 0.0
                for element in elements:

                    try:
                        value += self.points[element][idx]
                    except IndexError as e:
                        continue # means self.points deques aren't filled completely yet

                if self.combination == 'cumulative_force':
                    cumulative_points.append(value / len(elements))
                else:
                    cumulative_points.append(value)

            p = max(cumulative_points)

        else:
            maxes = []
            for element in elements:
                if len(self.points[element]):
                    maxes.append(max(self.points[element]))

            p = max(maxes)

        if p > 0:
            return 1.0 / p
        return 0.0


    def get_points_scaled(self, element, elements=None):

        if elements is None:
            elements = self.elements

        scale_factor = self.get_scale_factor(elements)
        if scale_factor == 0.0:
            return [0.0 for _ in range(0, len(self.points[element]))]

        r = []
        for amplitude in self.points[element]:
            r.append(amplitude * scale_factor)

        return r


    def draw_grid(self, context, elements=None):
      
        if elements is None:
            elements = self.elements
        
        scale_factor = self.get_scale_factor(elements)

        context.set_line_width(1)
        context.set_source_rgba(*self.colors['grid_minor'].tuple_rgba())
        #context.set_dash([1,1])

        for x in range(int(self.x + self.margin_left + self.padding_left), int(self.x + self.margin_left + self.padding_left + self.inner_width), int(self.step)):
            context.move_to(x, self.y + self.margin_top + self.padding_top)
            context.line_to(x, self.y + self.margin_top + self.padding_top + self.grid_height)
        
        context.stroke()
        
        if not self.autoscale:

            for i in range(0, 110, 10): # 0,10,20..100
                
                value = i / 100.0
                y = self.y + self.margin_top + self.padding_top + self.grid_height - self.grid_height * value

                context.move_to(self.x + self.margin_left + self.padding_left, y)
                context.line_to(self.x + self.margin_left + self.padding_left + self.inner_width, y)

            context.stroke()

        elif scale_factor > 0:
            
            if scale_factor > 1000:
                return # current maximum value under 1 permill, thus no guides are placed

            elif scale_factor > 100:
                # current maximum under 1 percent, place permill guides
                context.set_source_rgba(*self.colors['grid_milli'].tuple_rgba())
                for i in range(0, 10):
                    # place lines for 0-9 percent
                    value = i / 1000.0 * scale_factor
                    y = self.y + self.margin_top + self.padding_top + self.grid_height - self.grid_height * value

                    if y < self.y + self.margin_top + self.padding_top:
                        break # stop the loop if guides would be placed outside the gauge

                    context.move_to(self.x + self.margin_left + self.padding_left, y)
                    context.line_to(self.x + self.margin_left + self.padding_left + self.inner_width, y)
                
                context.stroke()

            elif scale_factor > 10:

                context.set_source_rgba(*self.colors['grid_minor'].tuple_rgba())
                for i in range(0, 10):
                    # place lines for 0-9 percent
                    value = i / 100.0 * scale_factor
                    y = self.y + self.margin_top + self.padding_top + self.grid_height - self.grid_height * value

                    if y < self.y + self.margin_top + self.padding_top:
                        break # stop the loop if guides would be placed outside the gauge

                    context.move_to(self.x + self.margin_left + self.padding_left, y)
                    context.line_to(self.x + self.margin_left + self.padding_left + self.inner_width, y)
                
                context.stroke()

            else: # major (10% step) guides
                context.set_source_rgba(*self.colors['grid_major'].tuple_rgba())
                for i in range(0, 110, 10): # 0,10,20..100
                    
                    value = i / 100.0 * scale_factor
                    y = self.y + self.margin_top + self.padding_top + self.grid_height - self.grid_height * value

                    if y < self.y + self.margin_top + self.padding_top:
                        break # stop the loop if guides would be placed outside the gauge

                    context.move_to(self.x + self.margin_left + self.padding_left, y)
                    context.line_to(self.x + self.margin_left + self.padding_left + self.inner_width, y)

                context.stroke()

        #context.set_dash([1,0]) # reset dash


    def draw_plot(self, context, points, colors, offset=None):

        coords = []

        for idx, amplitude in enumerate(points):

            if offset:
                amplitude += offset[idx]

            coords.append((
                self.x + idx * self.step + self.margin_left + self.padding_left,
                self.y + self.margin_top + self.padding_top + self.inner_height - (self.inner_height * amplitude)
            ))
      
       
        if self.line:

            # draw lines

            context.set_source_rgba(*colors['plot_line'].tuple_rgba())
            context.set_line_width(2)
            #context.set_line_cap(cairo.LINE_CAP_BUTT)

            for idx, (x, y) in enumerate(coords):
                if idx == 0:
                    context.move_to(x, y)
                else:
                    context.line_to(x, y)

            context.stroke()

        if self.pattern:

            context.set_source_surface(self.pattern(colors['plot_fill']))
            context.get_source().set_extend(cairo.Extend.REPEAT)
            
            context.move_to(self.x + self.margin_left + self.padding_left, self.y + self.margin_top + self.padding_top + self.inner_height)
            for idx, (x, y) in enumerate(coords):
                context.line_to(x, y)

            if offset: # "cut out" the offset at the bottom

                previous_amplitude = None
                for i, amplitude in enumerate(reversed(offset)):

                    if len(offset) - i > len(points):
                        continue # ignore x coordinates not reached yet by the graph

                    if (amplitude != previous_amplitude or i == len(offset) - 1):

                        offset_x = self.x + self.margin_left + self.padding_left + self.inner_width - i * self.step
                        offset_y = self.y + self.margin_top + self.padding_top + self.inner_height - self.inner_height * amplitude

                        context.line_to(offset_x, offset_y)

            else:
                context.line_to(x, self.y + self.margin_top + self.padding_top + self.inner_height)

            context.close_path()

            context.fill()

        if self.markers:

            # place points
            for (x, y) in coords:

                context.set_source_rgba(*colors['plot_marker'].tuple_rgba())

                context.arc(
                    x,
                    y,
                    2,
                    0,
                    2 * math.pi
                )
            
                context.fill()


    def update(self, context):
        
        for element in set(self.elements): # without set() the same element multiple times leads to multiple same points added every time.
            self.points[element].append(self.monitor.normalize(element))

        super(Plot, self).update(context)


    def draw(self, context):

        self.draw_background(context)

        if self.grid:
            self.draw_grid(context)

        if self.autoscale:
            scale_factor = self.get_scale_factor()
            if scale_factor == 0.0:
                text = u"∞X"
            else:
                text = "%sX" % pretty_si(self.get_scale_factor())
            self.app.draw_text(context, text, self.x + self.margin_left +self.padding_left + self.inner_width, self.y, align='right_top', color=self.colors['caption_scale'], font_size=self.get_style('font_size', 'scale'))

        colors_plot_marker = self.palette(self.colors['foreground'], len(self.elements))
        colors_plot_line = self.palette(self.colors['plot_line'], len(self.elements))
        colors_plot_fill = self.palette(self.colors['plot_fill'], len(self.elements))

        offset = [0.0 for _ in range(0, self.num_points)]
        for idx, element in enumerate(self.elements):

            colors = {
                'plot_marker': self.colors_plot_marker[idx],
                'plot_line': self.colors_plot_line[idx],
                'plot_fill': self.colors_plot_fill[idx]
            }

            if self.autoscale:
                points = self.get_points_scaled(element)
            else:
                points = list(self.points[element])

            if self.combination == 'separate':
                self.draw_plot(context, points, colors)

            elif self.combination.startswith('cumulative'):

                if self.combination == 'cumulative_force':
                    for idx in range(0, len(points)):
                        points[idx] /= len(self.elements)

                self.draw_plot(context, points, colors, offset)

                for idx, value in enumerate(points):
                    offset[idx] += value

        self.draw_captions(context)


class MirrorPlot(Plot):

    def __init__(self, app, monitor, scale_lock=True, **kwargs):

        self.up = kwargs['elements'][0]
        self.down = kwargs['elements'][1]
        kwargs['elements'] = self.up + self.down

        super(MirrorPlot, self).__init__(app, monitor, **kwargs)
        self.y_center = self.y + self.margin_top + self.padding_top + (self.inner_height / 2)
        self.scale_lock = scale_lock # bool, whether to use the same scale for up and down

        self.grid_height /= 2

        palette_len = max((len(self.up), len(self.down)))
        self.colors_plot_marker = self.palette(self.colors['foreground'], palette_len)
        self.colors_plot_line = self.palette(self.colors['plot_line'], palette_len)
        self.colors_plot_fill = self.palette(self.colors['plot_fill'], palette_len)


    def legend_info(self):

        data = collections.OrderedDict()
      
        colors = self.palette(self.colors['foreground'], max(len(self.up), len(self.down)))

        for elements in (self.up, self.down):
            for idx, element in enumerate(elements):
                data[element] = (colors[idx], element)

        return data

    def prepare_points(self):

        self.points = collections.OrderedDict()

        for element in self.elements:
            self.points[element] = collections.deque([], self.num_points)


    def get_scale_factor(self, elements=None):


        if self.scale_lock:

            scale_up = super(MirrorPlot, self).get_scale_factor(self.up)
            scale_down = super(MirrorPlot, self).get_scale_factor(self.down)

            if (scale_up > 0) and (scale_down > 0): # both values > 0
                return min((scale_up, scale_down))

            elif (scale_up == 0) != (scale_down == 0): # one value is 0
                return max(scale_up, scale_down)

            else: # both values are 0
                return 0


        return super(MirrorPlot, self).get_scale_factor(elements)


    def draw_grid(self, context, elements=None):
        
        context.set_line_width(1)
        context.set_source_rgba(*self.colors['grid_milli'].tuple_rgba())

        context.move_to(self.x + self.margin_left + self.padding_left, self.y_center)
        context.line_to(self.x + self.margin_left + self.padding_left + self.inner_width, self.y_center)
        context.stroke()

        if elements == self.up:
            context.translate(0, self.grid_height)
            super(MirrorPlot, self).draw_grid(context, elements=elements)
            context.translate(0, -self.grid_height)

        else:
            super(MirrorPlot, self).draw_grid(context, elements=elements)


    def draw_plot(self, context, points, colors, offset=None):

        points = [x/2 for x in points]

        if not offset is None:
            offset =[x/2 for x in offset]

        context.translate(0, -self.inner_height / 2)
        super(MirrorPlot, self).draw_plot(context, points, colors, offset=offset)
        context.translate(0, self.inner_height / 2)


    def draw_plot_negative(self, context, points, colors, offset=None):

        points = [-x for x in points]

        if not offset is None:
            offset = [-x for x in offset]
        self.draw_plot(context, points, colors, offset=offset)


    def update(self, context):

        for element in set(self.up + self.down):

            self.points[element].append(self.monitor.normalize(element))

        super(Plot, self).update(context) # TRAP: calls parent of parent, not direct parent!


    def draw(self, context):

        # TODO: This is mostly a copy-paste of Plot.update+draw, needs moar DRY.

        self.draw_background(context)

        for elements, drawer in ((self.up, self.draw_plot), (self.down, self.draw_plot_negative)):
          
            if self.grid:
                self.draw_grid(context, elements)

            if self.autoscale:
                scale_factor = self.get_scale_factor(elements)
                if scale_factor == 0.0:
                    text = u"∞X"
                else:
                    text = "%sX" % pretty_si(self.get_scale_factor(elements))

                if elements == self.up:
                    self.app.draw_text(context, text, self.x + self.margin_left + self.padding_left + self.inner_width, self.y + self.margin_top + self.padding_top, align='right_bottom', color=self.colors['caption_scale'], font_size=10)
                elif not self.scale_lock: # don't show 'down' scalefactor if it's locked
                    self.app.draw_text(context, text, self.x + self.margin_left + self.padding_left + self.inner_width, self.y + self.margin_top + self.padding_top + self.inner_height, align='right_top', color=self.colors['caption_scale'], font_size=10)

            offset = [0.0 for _ in range(0, self.num_points)]
            for idx, element in enumerate(elements):

                colors = {
                    'plot_marker': self.colors_plot_marker[idx],
                    'plot_line': self.colors_plot_line[idx],
                    'plot_fill': self.colors_plot_fill[idx]
                }

                if self.autoscale:
                    points = self.get_points_scaled(element, elements)
                else:
                    points = list(self.points[element])

                if self.combination == 'separate':
                    drawer(context, points, colors)

                elif self.combination.startswith('cumulative'):

                    if self.combination == 'cumulative_force':
                        for idx in range(0, len(points)):
                            points[idx] /= len(elements)

                    drawer(context, points, colors, offset)

                    for idx, value in enumerate(points):
                        offset[idx] += value

        self.draw_captions(context)


class Box(object):

    """
        Can wrap multiple Gauges, used for layouting.
        Orders added gauges from left to right and top to bottom.
    """

    def __init__(self, app, x, y, width, height):

        self._last_right = 0
        self._last_top = 0
        self._last_bottom = 0 

        self.app = app
        self.x = x
        self.y = y
        self.width = width
        self.height = height


    def place(self, component, cls, **kwargs):
    
        width = kwargs.get('width', None)
        height = kwargs.get('height', None)

        if width is None:
            width = self.width - self._last_right

            if height is None:
                height = self._last_bottom - self._last_top # same height as previous gauge

                if height == 0: # should only happen on first gauge
                    height = self.height

        elif height is None:
            height = self.height - self._last_bottom

        if self._last_right + width > self.width: # move to next "row"
            x = 0
            y = self._last_bottom

        else:
            x = self._last_right
            y = self._last_top

        kwargs['x'] = self.x + x
        kwargs['y'] = self.y + y
        kwargs['width'] = width
        kwargs['height'] = height
        
        self.app.add_gauge(component, cls, **kwargs)

        self._last_right = x + width
        self._last_top = y
        self._last_bottom = y + height


class Gulik(object):

    monitor_table = {
        'cpu': CPUMonitor,
        'memory': MemoryMonitor,
        'network': NetworkMonitor,
        'battery': BatteryMonitor
    }


    def __init__(self, configpath):

        self.started = False
        self.will_to_live = True # only needed because of gtk

        self.screen = Gdk.Screen.get_default()
        self.window = Window()
        self.window.connect('draw', self.draw)

        self.configpath = configpath
        self.config = {}
        self.setup = self.autosetup

        self.monitors = {}
        self.gauges = []
        #self.boxes = []

        self.apply_config()

    def reset(self):

        # clear out any existing gauges and monitors
        for monitor in self.monitors.values(): # kill any existing monitors
            monitor.seppuku = True

        self.monitors = {}
        self.gauges = []
        #self.boxes = []


    def module_to_config(self, locals):

        config = {}

        for key in DEFAULTS:
            config[key] = locals.get(key, DEFAULTS[key])

        for key in set(locals) - set(DEFAULTS): # iterates through everything defined in config.py we haven't already covered with DEFAULTS
            if key == key.upper(): # all-caps means it's config
                config[key] = locals[key]

        if not 'COLOR_FOREGROUND_TEXT' in config:
            config['COLOR_FOREGROUND_TEXT'] = config['COLOR_CAPTION']

        return config


    def add_netdata_monitors(self):

        # create monitors for all netdata hosts
        for host in self.config['NETDATA_HOSTS']:

            if isinstance(host, (list, tuple)): # handle (host, port) tuples and bare hostnames as values
                host, port = host
            else:
                port = 19999

            component = f"netdata-{host}"
            self.monitors[component] = NetdataMonitor(self, component, host, port)


    def resize_and_move(self):

        self.window.resize(self.config['WIDTH'], self.config['HEIGHT'])
        self.window.move(self.config['X'], self.config['Y']) # move apparently must be called after show_all


    def apply_config(self):

        print(f"Trying to load config from {self.configpath}…")

        custom = False # whether gulik is using a custom configuration
        old_config = self.config
        old_setup = self.setup

        self.config = {}
        try:
            fd = open(self.configpath, mode='r')
            config_string = fd.read()
        except OSError as e:
            print("No config at '%s' or insufficient permissions to read it. Falling back to defaults…" % self.configpath)
            config_locals = DEFAULTS

        else:
            try:
                config_locals = {}
                exec(config_string, config_locals)
                custom = True
            except Exception as e:
                print("Error in '%s': %s" % (self.configpath, e))
                print("Falling back to defaults…")
                config_locals = DEFAULTS

        self.config = self.module_to_config(config_locals)
        self.resize_and_move()

        if 'setup' in config_locals:
            self.setup = functools.partial(config_locals['setup'], app=self)
        else:
            self.setup = self.autosetup

        self.reset()

        self.add_netdata_monitors()


        # NOTE: psutil-based monitors are autoloaded in add_gauge and thus don't have to be handled here like netdata monitors

        # finally, run the actual setup function to place all gauges
        try:
            self.setup()
            print("Done.")

        except Exception as e:

            # Hokay, so I *think* this catches all failure scenarios for config (re)loads…
            
            if not custom:
                print("Sorry, friend. The autosetup seems to be broken on your machine. You might want to file a bug report.")
                print("Alternatively, you can create a custom-fit configuration at ~/.config/gulik/config.py.")
                self.stop()
                return

            else:

                if isinstance(self.setup, functools.partial):
                    name = self.setup.func.__name__
                else:
                    name = self.setup.__name__
                print(f"Your custom 'setup' function failed - {type(e).__name__}: {e}")

                print("Falling back to previous configuration")
                self.monitors = {}
                self.gauges = []
                self.config = old_config or self.module_to_config(DEFAULTS) # DEFAULTS if config was empty before (happens on first time this function is called on a Gulik object
                self.setup = old_setup
                self.resize_and_move()
                self.add_netdata_monitors()

                try:
                    self.setup()
                except Exception as e:
                    print("Previous setup failed, too. Reverting to autosetup.")
                    self.monitors = {}
                    self.gauges = []
                    self.config = self.module_to_config(DEFAULTS)
                    self.setup = self.autosetup
                    self.resize_and_move()
                    self.add_netdata_monitors()
                    try:
                        self.setup()
                    except Exception as e:
                        raise
                        print("Well, this should never happen.")
                        print(e)
                        self.stop()


    def signal_reload(self, *_):

        self.apply_config() # re-creates monitors and gauges

        for monitor in self.monitors.values():
            monitor.start()


    def tick(self):

        for monitor in self.monitors.values():
            monitor.tick()
        self.window.queue_draw()
        return self.will_to_live # gtk stops executing timeout callbacks if they don't return True


    def draw_text(self, context, text, x, y, align=None, color=None, font_size=None, font_weight=None):
        
        if align is None:
            align = 'left_top'
        
        if color is None:
            color = self.config['COLOR_CAPTION']
        
        context.set_source_rgba(*color.tuple_rgba())

        font_size = font_size or self.config['FONT_SIZE']
        font_weight = font_weight or self.config['FONT_WEIGHT']

        font = Pango.FontDescription('%s %s %d' % (self.config['FONT'], font_weight, font_size))

        layout = PangoCairo.create_layout(context)
        layout.set_font_description(font)
        layout.set_text(text, -1)
        
        size = layout.get_pixel_size()

        x_offset, y_offset = alignment_offset(align, size)
        
        context.translate(x + x_offset, y + y_offset)

        PangoCairo.update_layout(context, layout)
        PangoCairo.show_layout(context, layout)

        context.translate(-x - x_offset, -y - y_offset)

        return size

    
    def draw_gulik(self, context):

        # gulik is the messenger of our lady discordia.
        # he is 200 by 133 poxels big
        gulik = cairo.ImageSurface.create_from_png(os.path.join(__path__[0], 'gulik.png'))
        context.save()
        context.set_operator(cairo.OPERATOR_SOFT_LIGHT)
        context.translate(self.config['WIDTH'] - 200, self.config['HEIGHT'] - 133)
        context.set_source_surface(gulik)
        context.rectangle(0, 0, 200, 133)
        context.fill()
        context.restore()


    def draw(self, window, context):

        context.set_operator(cairo.OPERATOR_CLEAR)
        context.paint()
        context.set_operator(cairo.OPERATOR_OVER)

        context.set_source_rgba(*self.config['COLOR_WINDOW_BACKGROUND'].tuple_rgba())
        context.rectangle(0, 0, self.config['WIDTH'], self.config['HEIGHT'])
        context.fill()

        self.draw_gulik(context)

        for gauge in self.gauges:
            gauge.update(context)


    def start(self):

        assert self.config['FPS'] != 0
        assert isinstance(self.config['FPS'], float) or self.config['FPS'] >= 1

        for monitor in self.monitors.values():
            monitor.start()

        signal.signal(signal.SIGINT, self.stop) # so ctrl+c actually kills gulik
        signal.signal(signal.SIGTERM, self.stop) # so kill actually kills gulik, and doesn't leave a braindead Gtk.main and Collector processes dangling around
        signal.signal(signal.SIGUSR1, self.signal_reload) # reload config on user-defined signal. (no, not HUP)
        GLib.timeout_add(1000/self.config['FPS'], self.tick)
        self.tick() # first tick without delay so we get output asap
        self.started = True
        Gtk.main() # blocks until Gtk.main.quit is called
        print("\nThank you for flying with phryk evil mad sciences, LLC. Please come again.")


    def stop(self, num=None, frame=None):

        spinner = '▏▎▍▌▋▊▉▉▊▋▌▍▎'
        self.will_to_live = False # will stop calls by gtk main loop to self.tick

        for monitor in self.monitors.values():
            monitor.seppuku = True
            i = 0

            while monitor.is_alive(): # show spinner animation until the monitor is finished

                i += 1

                spinner_idx = i % len(spinner)

                sys.stdout.write(f"\r{spinner[spinner_idx]}    ")
                sys.stdout.flush()
                time.sleep(1/24)

            #sys.stdout.write("\rn")
            #sys.stdout.flush()

        if self.started: # Gtk.main.quit throws an error if called when not started
            Gtk.main_quit()
        else:
            exit() # to actually quit the program before Gtk.main is running


    def box(self, x=0, y=0, width=None, height=None): 

        width = ignore_none(width, self.config['WIDTH'] - x)
        height = ignore_none(height, self.config['HEIGHT'] - y)

        box = Box(self, x, y, width, height)
        #self.boxes.append(box)
        return box


    def add_gauge(self, component, cls, **kwargs):

        if not component in self.monitors:
            if component in self.monitor_table:
                print("Autoloading %s!" % self.monitor_table[component].__name__)
                self.monitors[component] = self.monitor_table[component](self, component)
            elif component.startswith('netdata-'):
                raise LookupError(f"Unknown netdata host '{component[8:]}'")
            else:
                raise LookupError("No monitor class known for component '%s'. Custom monitor classes have to be added to Gulik.monitor_table to enable autoloading." % component)

        monitor = self.monitors[component]

            
        gauge = cls(self, monitor, **kwargs)
        self.gauges.append(gauge)


    def autosetup(self, x=0, y=0, width=None, height=None):

        box = self.box(x, y, width, height)

        cpu_num = psutil.cpu_count()
        all_cores = ['core_%d' % x for x in range(0, cpu_num)]


        box.place(
            'cpu',
            Arc,
            #elements=['aggregate'],
            elements=all_cores,
            width=box.width, 
            height=box.width,
            stroke_width=10,
            combination='cumulative_force',
            legend=False,
            padding_bottom=5,
            captions=[
                {
                    'text': '{aggregate:.1f}%',
                    'position': 'center_center',
                    'align': 'center_center',
                },
                {
                    'text': '{count} cores',
                    'position': 'right_bottom',
                    'align': 'right_bottom',
                }
            ]
        )

        box.place(
            'cpu',
            Plot,
            elements=all_cores,
            width=box.width,
            height=130,
            padding_bottom=50,
            autoscale=True,
            combination='cumulative_force',
            markers=False
        )

        box.place(
            'memory',
            Arc,
            elements=['other', 'top_3', 'top_2', 'top_1'],
            width=box.width,
            height=box.width + 100,
            padding_bottom=80, # make space for 4 rows of legend
            stroke_width=30,
            combination='cumulative',
            #legend_format="{{{element}}}",
            legend_order='reverse',
            legend_format="{{{element}.name}} ({{{element}.private}})",
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
                }
            ]
        )

        last_gauge = self.gauges[-1]
        palette = [color for color in reversed(last_gauge.palette(last_gauge.colors['foreground'], 4))]
        #box.place('memory', Text, text='{top_1.name} ({top_1.private})', width=box.width, height=25, foreground=palette[0]) 
        #box.place('memory', Text, text='{top_2.name} ({top_2.private})', width=box.width, height=25, foreground=palette[1]) 
        #box.place('memory', Text, text='{top_3.name} ({top_3.private})', width=box.width, height=25, foreground=palette[2]) 
        #box.place('memory', Text, text='other({other.private}/{other.count})', width=box.width, height=25, foreground=palette[3]) 

        all_nics = [x for x in psutil.net_if_addrs().keys()]
        all_nics_up = ['%s.bytes_sent' % x for x in all_nics]
        all_nics_down = ['%s.bytes_recv' % x for x in all_nics]
        all_nics_up_packets_ = ['%s.packets_sent' % x for x in all_nics]
        all_nics_down_packets = ['%s.packets_recv' % x for x in all_nics]
        all_nics_up_errors = ['%s.errout' % x for x in all_nics]
        all_nics_down_errors = ['%s.errin' % x for x in all_nics]
        all_nics_up_drop = ['%s.dropout' % x for x in all_nics]
        all_nics_down_drop = ['%s.dropin' % x for x in all_nics]

        box.place(
            'network',
            MirrorArc,
            width=box.width,
            height=box.width,
            padding_bottom=5,
            legend=False,
            elements=[all_nics_up, all_nics_down],
            combination='cumulative_force',
            captions=[
                {
                    'text': '{aggregate.counters.bytes_sent}\n{aggregate.counters.bytes_recv}',
                    #'text': '{em0.all_addrs}',
                    'position': 'center_center',
                    'align': 'center_center',
                }
            ]
        )

        box.place(
            'network',
            MirrorPlot,
            width=box.width,
            height=130,
            elements=[all_nics_up, all_nics_down],
            markers=False,
            combination='cumulative_force'
        )

        #alignments = ['left_top', 'center_top','right_top']
        #palette = self.gauges[-1].colors_plot_marker
        #for idx, if_name in enumerate(all_nics):
        #    # build a legend
        #    color = palette[idx]
        #    align = alignments[idx % 3] # boom.
        #    box.place('network', Text, text=if_name, foreground=color, width=box.width/3, height=25, align=align)


        if psutil.sensors_battery() is not None:

            box.place('battery', Rect, width=box.width, height=60, elements=['battery'], captions=[
                    {
                        'text': '{percent}%',
                        'position': 'left_center',
                        'align': 'left_center',
                        'color': self.config['COLOR_FOREGROUND'],
                        'operator': Operator.DIFFERENCE, # fake cut-out effect, can break with custom configs
                    },
                    {
                        'text': '{state}',
                        'position': 'right_center',
                        'align': 'right_center',
                        'color': self.config['COLOR_FOREGROUND'],
                        'operator': Operator.DIFFERENCE,
                    },
                ]
            )
