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
    Return a human readable representation in bits given a size in bytes.
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




## FILLS ##
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


# CONFIG: TODO: Move into its own file, obvsly

DEFAULTS = {
    'FPS': 1,
    'COLORS': {
        'window_background': Color(0,0,0, 0.6),
        'gauge_background': Color(1,1,1, 0.1),
        'highlight': Color(0.5, 1, 0, 0.6),
        'text': Color(1,1,1, 0.6),
        'text_minor': Color(1,1,1, 0.3)
    },
    'PALETTE': functools.partial(palette_hue, distance=-120), # mhh, curry…
    'FONT': 'Orbitron',
    'FONT_WEIGHT': 'Light',
    'WIDTH': 200,
    'HEIGHT': Gdk.Screen().get_default().get_height(),
    'X': 0,
    'Y': 0,
    'NETDATA_HOSTS': []
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

    def __init__(self, queue_update, queue_data):

        super(Collector, self).__init__()
        self.daemon = True
        self.queue_update = queue_update
        self.queue_data = queue_data
        self.elements = []


    def terminate(self):

        self.queue_update.close()
        self.queue:data.close()
        os.kill(self.pid, signal.SIGKILL) # Would've done it cleaner, but after half a day of chasing some retarded quantenbug I'm done with this shit. Just nuke the fucking things from orbit.
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
        
        time.sleep(0.1) # according to psutil docs, there should at least be 0.1 seconds between calls to cpu_percent without interval


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
                            # FIXME: startswith('[') might make this BSD-specific
                            if  mmap.path.startswith('['): # assuming everything with a real path is "not really in ram", but no clue.
                                private += mmap.private * PAGESIZE
                                resident += mmap.rss * PAGESIZE
                                #shared += (mmap.rss - mmap.private) * PAGESIZE # FIXME: obviously broken, can yield negative values
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
                    pass#print("memory_maps failed!", e)

        info = DotDict({
            'total': vmem.total,
            'percent': total_use / vmem.total * 100,
            'available': vmem.total - total_use
        })

        processes_sorted = sorted(processes, key=lambda x: x['private'], reverse=True)

        for i, process in enumerate(processes_sorted[:3]):
            info['top_%d' % (i + 1)] = process 

        info['other'] = DotDict({
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

    def __init__(self, queue_update, queue_data, host, port):

        super(NetdataCollector, self).__init__(queue_update, queue_data)
        self.client = netdata.Netdata(host, port=port)


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
            data = self.client.data(chart, points=1, after=-1, options=['absolute']) # get the last second of data condensed to one point
        except netdata.NetdataException:
            pass
        else:
            self.queue_data.put((chart, data), block=True)


## Monitors ##

class Monitor(threading.Thread):

    collector_type = Collector

    def __init__(self, app):

        super(Monitor, self).__init__()
        self.app = app
        self.daemon = True
        self.seppuku = False

        self.queue_update = multiprocessing.Queue(1)
        self.queue_data = multiprocessing.Queue(1)
        self.collector = self.collector_type(self.queue_update, self.queue_data)
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
            #data = self.queue_data.get(block=True) # get new data from the collector as soon as it's available

            try:
                self.data = self.queue_data.get(timeout=1)
            except queue.Empty:
                pass # try again, but give thread the ability to die without waiting on collector indefinitely

        self.commit_seppuku()


    def commit_seppuku(self):

        print(f"{self.__class__.__name__} committing glorious seppuku!")

        self.queue_data.close()
        self.queue_update.close()
        self.collector.terminate()
        #print(f"TERMINATED {self.collector.pid}")
        self.collector.join()
        #print(f"JOINED {self.collector.pid}")


    def normalize(self, element=None):
        raise NotImplementedError("%s.normalize not implemented!" % self.__class__.__name__)


    def caption(self, fmt):
        raise NotImplementedError("%s.caption not implemented!" % self.__class__.__name__)


class CPUMonitor(Monitor):

    collector_type = CPUCollector

    def normalize(self, element=None):

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

    def normalize(self, element=None):

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
            if k != 'other':
                data[k]['name'] = self.data[k]['name']
            else:
                data[k]['count'] = self.data[k]['count']
            data[k]['private'] = pretty_bytes(self.data[k]['private'])
            data[k]['shared'] = pretty_bytes(self.data[k]['shared'])
        return fmt.format(**data)


class NetworkMonitor(Monitor):

    collector_type = NetworkCollector

    def __init__(self, app):

        super(NetworkMonitor, self).__init__(app)

        self.interfaces = collections.OrderedDict()

        if self.app.config['FPS'] < 2:
            deque_len = 2 # we need a minimum of 2 so we can get a difference
        else:
            deque_len = self.app.config['FPS'] # max size equal fps means this holds data of only the last second

        keys = ['bytes_sent', 'bytes_recv', 'packets_sent', 'packets_recv', 'errin', 'errout', 'dropin', 'dropout']

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

            #data = self.queue_data.get(block=True) # get new data from the collector as soon as it's available
            try:
                self.data = self.queue_data.get(timeout=1)
            except queue.Empty:
                pass # try again, but give thread the ability to die without waiting on collector indefinitely
            
            aggregates = {}
            for key in self.aggregate['counters']:
                #self.aggregate['counters'][k] = []
                aggregates[key] = 0

            self.aggregate['speed'] = 0
            for if_name, if_data in self.interfaces.items():
               
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
            get a specified count for a given interface as calculated for the last second
            EXAMPLE: self.count_sec('eth0', 'bytes_sent') will return count of bytes sent in the last second
        """

        if interface == 'aggregate':
            deque = self.aggregate['counters'][key]
        else:
            deque = self.interfaces[interface]['counters'][key]
        
        if self.app.config['FPS'] < 2:
            return (deque[-1] - deque[0]) / self.app.config['FPS'] # fps < 1 means data covers 1/fps seconds

        else:
            return deque[-1] - deque[0] # last (most recent) minus first (oldest) item


    def normalize(self, element=None):

        if_name, key = element.split('.')

        if if_name == 'aggregate':
            if len(self.aggregate['counters'][key]) >= 2:
                link_quality = float(self.aggregate['speed'] * 1024**2)
                return (self.count_sec(if_name, key) * 8) / link_quality

        elif len(self.interfaces[if_name]['counters'][key]) >= 2:
            link_quality = float(self.interfaces[if_name]['stats']['speed'] * 1024**2)

            return (self.count_sec(if_name, key) * 8) / link_quality

        return 0 # only happens if we have less than 2 datapoints in which case we can't establish used bandwidth


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

    def normalize(self, element=None):

        if not self.data:
            return 0

        return self.data.percent / 100.0


    def caption(self, fmt):

        if not self.data:
            return fmt

        data = self.data._asdict()

        if not data['power_plugged']:
            data['state'] = 'Draining'
        elif data['percent'] == 100:
            data['state'] = 'Full'
        else:
            data['state'] = 'Charging'

        return fmt.format(**data)


class NetdataMonitor(Monitor):

    collector_type = NetdataCollector

    def __init__(self, app, host, port):

        self.collector_type = functools.partial(self.collector_type, host=host, port=port)

        super(NetdataMonitor, self).__init__(app)

        self.charts = set()
        self.normalization_values = {} # keep a table of known maximums because netdata doesn't supply absolute normalization values
        self.last_updates = {}
        
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
                self.last_updates[chart] = 0

                if self.netdata_info:
                    if not chart in self.netdata_info['charts']:
                        raise ValueError(f"Invalid chart: {chart} on netdata instance {self.host}:{self.port}!")

                    chart_info = self.netdata_info['charts'][chart]
                    if chart_info['units'] == 'percentage':
                        self.normalization_values[chart] = 100
                    else:
                        self.normalization_values[chart] = 0

                    self.last_updates[chart] = chart_info['last_entry']

                self.charts.add(chart)

    
    def run(self):

        #while self.collector.is_alive():
        while not self.seppuku:
            #(chart, data) = self.queue_data.get(block=True) # get new data from the collector as soon as it's available
            try:
                (chart, data) = self.queue_data.get(timeout=1)
                self.data[chart] = data
            except queue.Empty:
                pass # try again

        self.commit_seppuku()


    def tick(self):
        if not self.queue_update.full():
        #if not self.seppuku: # don't request more updates to collector when we're trying to die
            for chart in self.charts:
                self.queue_update.put(f"UPDATE {chart}", block=True)


    def normalize(self, element=None):

        parts = element.split('.')

        chart = '.'.join(parts[:2])

        #if chart not in self.charts or not self.data[chart]:
        if not chart in self.data:
            print(f"No data for {chart}")
            return 0 #

        #timestamp = self.data[chart]['data'][0][0] # first element of a netdata datapoint is always time
        #if timestamp > self.last_updates[chart]:

        subelem = parts[2]
        subidx = self.data[chart]['labels'].index(subelem)
        value = self.data[chart]['data'][0][subidx]

        #return data/100
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

    def __init__(self, app, monitor, x=0, y=0, width=100, height=100, padding=5, elements=None, captions=None, foreground=None, background=None, pattern=None, palette=None, combination=None, operator=cairo.Operator.OVER):

        self.app = app
        self.monitor = monitor
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.padding = padding
        self.elements = elements if elements is not None else [None]
        self.captions = captions if captions else list()
        self.operator = operator

        self.monitor.register_elements(elements)

        self.colors = {}

        if foreground is None:
            self.colors['foreground'] = self.app.config['COLORS']['highlight']
        else:
            self.colors['foreground'] = foreground

        if background is None:
            self.colors['background'] = self.app.config['COLORS']['gauge_background']
        else:
            self.colors['background'] = background

        self.pattern = pattern
        self.palette = palette or self.app.config['PALETTE'] # function to generate color palettes with

        self.combination = combination or 'separate' # combination mode when handling multiple elements. 'separate', 'cumulative' or 'cumulative_force'. cumulative assumes all values add up to max 1.0, while separate assumes every value can reach 1.0 and divides all values by the number of elements handled



    @property
    def inner_width(self):
        return self.width - 2 * self.padding


    @property
    def inner_height(self):
        return self.height - 2 * self.padding


    def set_brush(self, context, color):

        if self.pattern:
            context.set_source_surface(self.pattern(color))
            context.get_source().set_extend(cairo.Extend.REPEAT)

        else:
            context.set_source_rgba(*color.tuple_rgba())


    def draw_background(self, context):

        context.set_source_rgba(*self.colors['background'].tuple_rgba())
        context.rectangle(self.x + self.padding, self.y + self.padding, self.inner_width, self.inner_height)
        context.fill()


    def draw_captions(self, context):

        for caption in self.captions:

            if 'operator' in caption:
                context.save()
                context.set_operator(caption['operator'])

            if 'position' in caption:

                if isinstance(caption['position'], str):
                    # handle alignment-style strings like "center_bottom"
                    position = [-x for x in alignment_offset(caption['position'], (self.width - 2 * self.padding, self.height - 2 * self.padding))]

                else:
                    position = caption['position']

            else:
                position = [0, 0]

            position = [position[0] + self.x + self.padding, position[1] + self.y + self.padding]

            caption_text = self.monitor.caption(caption['text'])

            self.app.render_text(context, caption_text, position[0], position[1], align=caption.get('align', None), color=caption.get('color', None), font_size=caption.get('font_size', None))

            if 'operator' in caption:
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


class MarqueeGauge(Gauge):

    def __init__(self, app, monitor, text, speed=25, align=None, **kwargs):
       
        if 'foreground' not in kwargs:
            kwargs['foreground'] = self.app.config['COLORS']['text']

        super(MarqueeGauge, self).__init__(app, monitor, **kwargs)
        self.text = text # the text to be rendered, a format string passed to monitor.caption
        self.previous_text = '' # to be able to detect change

        self.speed = speed

        if align is None:
            align = 'left_top'
        self.align = align

        surface = cairo.ImageSurface(cairo.Format.ARGB32, 10, 10)
        context = cairo.Context(surface)
        font = Pango.FontDescription('%s %s %d' % (self.app.config['FONT'], self.app.config['FONT_WEIGHT'], 10))
        layout = PangoCairo.create_layout(context)
        layout.set_font_description(font)
        layout.set_text('0', -1) # naively assuming 0 is the highest glyph
        size = layout.get_pixel_size()
        if size[1] > 10:
            self.font_size = self.inner_height * 10/size[1]
        else:
            self.font_size = self.inner_height
        self.direction = 'left'
        self.offset = 0.0
        self.step = speed / self.app.config['FPS'] # i.e. speed in pixel/s


    def draw(self, context):
        
        text = self.monitor.caption(self.text)

        context.save()
        #context.rectangle(self.x + self.padding, self.y + self.padding, self.inner_width, self.inner_height)
        context.rectangle(self.x, self.y, self.width, self.height)
        context.clip()

        context.set_source_rgba(*self.colors['foreground'].tuple_rgba())
        font = Pango.FontDescription('%s %s %d' % (self.app.config['FONT'], self.app.config['FONT_WEIGHT'], self.font_size))

        layout = PangoCairo.create_layout(context)
        layout.set_font_description(font)
        layout.set_text(text, -1)
        
        size = layout.get_pixel_size()
        align_offset = alignment_offset(self.align, size) # needed for the specified text alignment
        max_offset = size[0] - self.inner_width # biggest needed offset for marquee, can be negative if all text fits

        if max_offset <= 0 or text != self.previous_text:
            self.direction = 'left'
            self.offset = 0

        x = self.x + self.padding + align_offset[0] - self.offset
        
        if self.align.startswith('center'):
            x += self.inner_width / 2

        elif self.align.startswith('right'):
            x += self.inner_width


        y = self.y + align_offset[1]

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


class RectGauge(Gauge):

    def draw_rect(self, context, value, color, offset=0.0):

            self.set_brush(context, color)
            
            context.rectangle(self.x + self.padding + self.inner_width * offset, self.y + self.padding, self.inner_width * value, self.inner_height)
            context.fill()


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


class MirrorRectGauge(Gauge):

    def __init__(self, app, monitor, **kwargs):

        self.left = kwargs['elements'][0]
        self.right = kwargs['elements'][1]
        kwargs['elements'] = self.left + self.right
        super(MirrorRectGauge, self).__init__(app, monitor, **kwargs)
        self.x_center = self.x + self.width / 2
        self.draw_left = self.draw_rect_negative
        self.draw_right = self.draw_rect

    
    def draw_rect(self, context, value, color, offset=0.0):

            self.set_brush(context, color)
            
            context.rectangle(self.x_center + self.inner_width / 2 * offset, self.y + self.padding, self.inner_width / 2 * value, self.inner_height)
            context.fill()


    def draw_rect_negative(self, context, value, color, offset=0.0):

            self.set_brush(context, color)
            
            context.rectangle(self.x_center - self.inner_width / 2 * offset - self.inner_width / 2 * value, self.y + self.padding, self.inner_width / 2 * value, self.inner_height)
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


class ArcGauge(Gauge):

    def __init__(self, app, monitor, stroke_width=5, **kwargs):

        super(ArcGauge, self).__init__(app, monitor, **kwargs)
        self.stroke_width = stroke_width
        self.radius = (min(self.width, self.height) / 2) - (2 * self.padding) - (self.stroke_width / 2)
        self.x_center = self.x + self.width / 2
        self.y_center = self.y + self.height / 2


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


class MirrorArcGauge(MirrorRectGauge, ArcGauge):

    def __init__(self, app, monitor, **kwargs):

        super(MirrorArcGauge, self).__init__(app, monitor, **kwargs)
        self.draw_left = self.draw_arc_negative
        self.draw_right = self.draw_arc


    def draw_arc(self, context, value, color, offset=0.0):

        value /= 2
        offset /= 2

        super(MirrorArcGauge, self).draw_arc(context, value, color, offset=offset)


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


class PlotGauge(Gauge):

    def __init__(self, app, monitor, num_points=None, autoscale=True, markers=True, line=True, grid=True, **kwargs):

        super(PlotGauge, self).__init__(app, monitor, **kwargs)

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

        #if self.combination == 'cumulative_force':
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

        for x in range(int(self.x + self.padding), int(self.x + self.padding + self.inner_width), int(self.step)):
            context.move_to(x, self.y + self.padding)
            context.line_to(x, self.y + self.padding + self.grid_height)
        
        context.stroke()
        
        if not self.autoscale:

            for i in range(0, 110, 10): # 0,10,20..100
                
                value = i / 100.0
                y = self.y + self.padding + self.grid_height - self.grid_height * value

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
                    y = self.y + self.padding + self.grid_height - self.grid_height * value

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
                    y = self.y + self.padding + self.grid_height - self.grid_height * value

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
                    y = self.y + self.padding + self.grid_height - self.grid_height * value

                    if y < self.y + self.padding:
                        break # stop the loop if guides would be placed outside the gauge

                    context.move_to(self.x + self.padding, y)
                    context.line_to(self.x + self.padding + self.inner_width, y)

                context.stroke()

        #context.set_dash([1,0]) # reset dash


    def draw_plot(self, context, points, colors, offset=None):

        coords = []

        for idx, amplitude in enumerate(points):

            if offset:
                amplitude += offset[idx]

            coords.append((
                self.x + idx * self.step + self.padding,
                self.y + self.padding + self.inner_height - (self.inner_height * amplitude)
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
            
            context.move_to(self.x + self.padding, self.y + self.padding + self.inner_height)
            for idx, (x, y) in enumerate(coords):
                context.line_to(x, y)

            if offset: # "cut out" the offset at the bottom

                previous_amplitude = None
                for i, amplitude in enumerate(reversed(offset)):

                    if len(offset) - i > len(points):
                        continue # ignore x coordinates not reached yet by the graph

                    if (amplitude != previous_amplitude or i == len(offset) - 1):

                        offset_x = self.x + self.padding + self.inner_width - i * self.step
                        offset_y = self.y + self.padding + self.inner_height - self.inner_height * amplitude

                        context.line_to(offset_x, offset_y)

            else:
                context.line_to(x, self.y + self.padding + self.inner_height)

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

        super(PlotGauge, self).update(context)


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
            self.app.render_text(context, text, self.x + self.padding + self.inner_width, self.y, align='right_top', color=self.colors['caption_scale'], font_size=10)
        

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


class MirrorPlotGauge(PlotGauge):

    #scale_lock = None # bool, whether to use the same scale for up and down

    def __init__(self, app, monitor, scale_lock=True, **kwargs):

        self.up = kwargs['elements'][0]
        self.down = kwargs['elements'][1]
        kwargs['elements'] = self.up + self.down

        super(MirrorPlotGauge, self).__init__(app, monitor, **kwargs)
        self.y_center = self.y + self.height / 2
        self.scale_lock = scale_lock
        self.grid_height /= 2

        palette_len = max((len(self.up), len(self.down)))
        self.colors_plot_marker = self.palette(self.colors['foreground'], palette_len)
        self.colors_plot_line = self.palette(self.colors['plot_line'], palette_len)
        self.colors_plot_fill = self.palette(self.colors['plot_fill'], palette_len)


    def prepare_points(self):

        self.points = collections.OrderedDict()

        for element in self.elements:
            self.points[element] = collections.deque([], self.num_points)


    def get_scale_factor(self, elements=None):


        if self.scale_lock:

            scale_up = super(MirrorPlotGauge, self).get_scale_factor(self.up)
            scale_down = super(MirrorPlotGauge, self).get_scale_factor(self.down)

            if (scale_up > 0) and (scale_down > 0): # both values > 0
                return min((scale_up, scale_down))

            elif (scale_up == 0) != (scale_down == 0): # one value is 0
                return max(scale_up, scale_down)

            else: # both values are 0
                return 0


        return super(MirrorPlotGauge, self).get_scale_factor(elements)


    def draw_grid(self, context, elements=None):
        
        context.set_line_width(1)
        context.set_source_rgba(*self.colors['grid_milli'].tuple_rgba())

        context.move_to(self.x + self.padding, self.y_center)
        context.line_to(self.x + self.padding + self.inner_width, self.y_center)
        context.stroke()

        if elements == self.up:
            context.translate(0, self.grid_height)
            super(MirrorPlotGauge, self).draw_grid(context, elements=elements)
            context.translate(0, -self.grid_height)

        else:
            super(MirrorPlotGauge, self).draw_grid(context, elements=elements)


    def draw_plot(self, context, points, colors, offset=None):

        points = [x/2 for x in points]

        if not offset is None:
            offset =[x/2 for x in offset]

        context.translate(0, -self.inner_height / 2)
        super(MirrorPlotGauge, self).draw_plot(context, points, colors, offset=offset)
        context.translate(0, self.inner_height / 2)


    def draw_plot_negative(self, context, points, colors, offset=None):

        points = [-x for x in points]

        if not offset is None:
            offset = [-x for x in offset]
        self.draw_plot(context, points, colors, offset=offset)


    def update(self, context):

        for element in set(self.up + self.down):

            self.points[element].append(self.monitor.normalize(element))

        super(PlotGauge, self).update(context) # TRAP: calls parent of parent, not direct parent!


    def draw(self, context):

        # TODO: This is mostly a copy-paste of PlotGauge.update+draw, needs moar DRY.

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
                    self.app.render_text(context, text, self.x + self.padding + self.inner_width, self.y + self.padding, align='right_bottom', color=self.colors['caption_scale'], font_size=10)
                elif not self.scale_lock: # don't show 'down' scalefactor if it's locked
                    self.app.render_text(context, text, self.x + self.padding + self.inner_width, self.y + self.padding + self.inner_height, align='right_top', color=self.colors['caption_scale'], font_size=10)
            


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
            x = self.x # 0 left offset
            y = self.y + self._last_bottom

        else:
            x = self.x + self._last_right
            y = self.y + self._last_top


        kwargs['x'] = x
        kwargs['y'] = y
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

        self.screen = Gdk.Screen.get_default()
        self.window = Window()
        self.window.connect('draw', self.draw)

        self.configpath = configpath
        self.config = {}
        self.setup = self.autosetup

        self.monitors = {}
        self.gauges = []
        self.boxes = []

        self.apply_config()


    def reset(self):

        for monitor in self.monitors.values(): # kill any existing monitors
            monitor.seppuku = True

        self.monitors = {}
        self.gauges = []
        self.boxes = []


    def apply_config(self):

        print(f"Trying to load config from {self.configpath}…")

        self.config = {}
        config_dict = {}
        try:
            fd = open(self.configpath, mode='r')
            config_string = fd.read()
        except OSError as e:
            print("No config at '%s' or insufficient permissions to read it. Falling back to defaults…" % self.configpath)

        else:
            try:
                exec(config_string, config_dict)
            except Exception as e:
                print("Error in '%s': %s" % (self.configpath, e))
                print("Falling back to defaults…")
                config_dict = DEFAULTS # because config_dict might be contaminated by half-done exec

        for key in DEFAULTS:
            self.config[key] = config_dict.get(key, DEFAULTS[key])

        self.window.resize(self.config['WIDTH'], self.config['HEIGHT'])
        self.window.move(self.config['X'], self.config['Y']) # move apparently must be called after show_all

        if 'setup' in config_dict:
            self.setup = functools.partial(config_dict['setup'], app=self)
        else:
            self.setup = self.autosetup

        self.reset() # clears out any existing gauges and monitors

        # create monitors for all netdata hosts
        for host in self.config['NETDATA_HOSTS']:

            if isinstance(host, (list, tuple)): # handle (host, port) tuples and bare hostnames as values
                host, port = host
            else:
                port = 19999

            self.monitors[f"netdata-{host}"] = NetdataMonitor(self, host, port)

        # NOTE: psutil-based monitors are autoloaded in add_gauge and thus don't have to be handled here like netdata monitors

        # finally, run the actual setup function to place all gauges
        self.setup()
        print("Done.")


    def signal_reload(self, *_):

        self.apply_config() # re-creates monitors and gauges

        for monitor in self.monitors.values():
            monitor.start()


    def tick(self):

        for monitor in self.monitors.values():
            monitor.tick()
        self.window.queue_draw()
        return True # gtk stops executing timeout callbacks if they don't return True


    def render_text(self, context, text, x, y, align=None, color=None, font_size=None):
        
        if align is None:
            align = 'left_top'
        
        if color is None:
            color = self.config['COLORS']['text']
        
        context.set_source_rgba(*color.tuple_rgba())

        if font_size is None:
            font_size = 12

        
        font = Pango.FontDescription('%s %s %d' % (self.config['FONT'], self.config['FONT_WEIGHT'], font_size))

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
        context.translate(0, self.window.height - 133)
        context.set_source_surface(gulik)
        context.rectangle(0, 0, 200, 133)
        context.fill()
        context.restore()


    def draw(self, window, context):

        context.set_operator(cairo.OPERATOR_CLEAR)
        context.paint()
        context.set_operator(cairo.OPERATOR_OVER)

        context.set_source_rgba(*self.config['COLORS']['window_background'].tuple_rgba())
        context.rectangle(0, 0, self.window.width, self.window.height)
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
        Gtk.main()
        print("\nThank you for flying with phryk evil mad sciences, LLC. Please come again.")


    def stop(self, num, frame):

        spinner = '▏▎▍▌▋▊▉▉▊▋▌▍▎'

        for monitor in self.monitors.values():
            monitor.seppuku = True
            i = 0
            while monitor.is_alive():
                #pass # wait for the thread to be dead
                i += 1

                spinner_idx = i % len(spinner)

                sys.stdout.write(f"\r{spinner[spinner_idx]}    ")
                sys.stdout.flush()
                time.sleep(1/24)

            #sys.stdout.write("\rn")
            #sys.stdout.flush()

        Gtk.main_quit()


    def box(self, x=0, y=0, width=None, height=None): 

        width = width if not width is None else self.window.width
        height = height if not height is None else self.window.height

        box = Box(self, x, y, width, height)
        self.boxes.append(box)
        return box


    def add_gauge(self, component, cls, **kwargs):

        if not component in self.monitors:
            if component in self.monitor_table:
                print("Autoloading %s!" % self.monitor_table[component].__name__)
                self.monitors[component] = self.monitor_table[component](self)
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
            ArcGauge,
            #elements=['aggregate'],
            elements=all_cores,
            width=box.width, 
            height=box.width,
            stroke_width=10,
            combination='cumulative_force',
            captions=[
                {
                    'text': '{aggregate:.1f}%',
                    'position': 'center_center',
                    'align': 'center_center',
                    'font_size': 14
                },
                {
                    'text': '{count} cores',
                    'position': 'right_bottom',
                    'align': 'right_bottom',
                    'color': self.config['COLORS']['text_minor'],
                    'font_size': 8
                }
            ]
        )

        #box.place('cpu', ArcGauge, elements=['core_0'], width=box.width / 4, height=box.width / 4)
        #box.place('cpu', ArcGauge, elements=['core_1'], width=box.width / 4, height=box.width / 4)
        #box.place('cpu', ArcGauge, elements=['core_2'], width=box.width / 4, height=box.width / 4)
        #box.place('cpu', ArcGauge, elements=['core_3'], width=box.width / 4, height=box.width / 4)
        box.place('cpu', PlotGauge, elements=all_cores, width=box.width, height=100, padding=15, pattern=stripe45, autoscale=True, combination='cumulative_force', markers=False)#, line=False, grid=False)
        #box.place('cpu', PlotGauge, elements=all_cores, width=box.width, height=100, padding=15, pattern=stripe45, autoscale=True, combination='separate', markers=False)#, line=False, grid=False)
        #box.place('cpu', RectGauge, elements=all_cores, width=box.width, height=50, padding=15, pattern=stripe45, combination='cumulative_force')

        box.place(
            'memory',
            ArcGauge,
            elements=['other', 'top_3', 'top_2', 'top_1'],
            width=box.width,
            height=box.width,
            stroke_width=30,
            combination='cumulative',
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
                    'color': self.config['COLORS']['text_minor'],
                    'font_size': 8,
                }
            ]
        )

        last_gauge = self.gauges[-1]
        palette = [color for color in reversed(last_gauge.palette(last_gauge.colors['foreground'], 4))]
        box.place('memory', MarqueeGauge, text='{top_1.name} ({top_1.private})', width=box.width, height=25, foreground=palette[0]) 
        box.place('memory', MarqueeGauge, text='{top_2.name} ({top_2.private})', width=box.width, height=25, foreground=palette[1]) 
        box.place('memory', MarqueeGauge, text='{top_3.name} ({top_3.private})', width=box.width, height=25, foreground=palette[2]) 
        box.place('memory', MarqueeGauge, text='other({other.private}/{other.count})', width=box.width, height=25, foreground=palette[3]) 

        all_nics = [x for x in psutil.net_if_addrs().keys()]
        all_nics_up = ['%s.bytes_sent' % x for x in all_nics]
        all_nics_down = ['%s.bytes_recv' % x for x in all_nics]
        all_nics_up_packets_ = ['%s.packets_sent' % x for x in all_nics]
        all_nics_down_packets = ['%s.packets_recv' % x for x in all_nics]
        all_nics_up_errors = ['%s.errout' % x for x in all_nics]
        all_nics_down_errors = ['%s.errin' % x for x in all_nics]
        all_nics_up_drop = ['%s.dropout' % x for x in all_nics]
        all_nics_down_drop = ['%s.dropin' % x for x in all_nics]

        box.place('network', MirrorArcGauge, width=box.width, height=box.width, elements=[all_nics_up, all_nics_down], combination='cumulative_force', captions=[
                {
                    'text': '{aggregate.counters.bytes_sent}\n{aggregate.counters.bytes_recv}',
                    #'text': '{em0.all_addrs}',
                    'position': 'center_center',
                    'align': 'center_center',
                }
            ]
        )

        box.place('network', MirrorPlotGauge, width=box.width, height=100, padding=15, elements=[all_nics_up, all_nics_down], pattern=stripe45, markers=False, combination='cumulative_force')#, scale_lock=False)

        alignments = ['left_top', 'center_top','right_top']
        palette = self.gauges[-1].colors_plot_marker
        for idx, if_name in enumerate(all_nics):
            # build a legend
            color = palette[idx]
            align = alignments[idx % 3] # boom.
            box.place('network', MarqueeGauge, text=if_name, foreground=color, width=box.width/3, height=25, padding=5, align=align)

        #box.place('network', MarqueeGauge, width=box.width, height=45, padding=15, text='🦆{lo0.counters.bytes_recv} AND SOMETHING TO MAKE IT SCROLL 💡')

        #box.place('network', MirrorPlotGauge, width=box.width, height=100, padding=15, elements=[['aggregate.bytes_sent'], ['aggregate.bytes_recv']], pattern=stripe45, markers=False)#, combination='cumulative_force')
        

        if psutil.sensors_battery() is not None:

            box.place('battery', RectGauge, width=box.width, height=60, padding=15, captions=[
                    {
                        'text': '{state}…',
                        'position': 'left_center',
                        'align': 'left_center',
                        'operator': cairo.Operator.OVERLAY,
                    },
                    {
                        'text': '{percent}%',
                        'position': 'right_center',
                        'align': 'right_center',
                        'operator': cairo.Operator.OVERLAY,
                    },
                ]
            )
