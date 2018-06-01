import time
import random
import signal
import threading
import multiprocessing
import psutil
import cairo
import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk


class PeriodicCall(threading.Thread):

    """ Periodically forces a window to redraw """

    target = None
    interval = None

    def __init__(self, target, interval):
        
        super(PeriodicCall, self).__init__()
        self.daemon = True

        self.target = target
        self.interval = interval


    def run(self):

        while True: # This thread will automatically die with its parent because of the daemon flag

            self.target()
            time.sleep(self.interval)


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


        self.connect('draw', self.tick)

        self.connect('delete-event', Gtk.main_quit)
        self.show_all()
        self.move(0, 32) # move apparently must be called after show_all


    def tick(self, wid, cr):

        cr.set_operator(cairo.OPERATOR_CLEAR)
        cr.paint()
        cr.set_operator(cairo.OPERATOR_OVER)

        cr.set_source_rgba(1, 0, 0, 0.4)
        cr.rectangle(0, 0, random.randint(0, self.width), random.randint(0, self.height))
        cr.fill()


    @property
    def width(self):

        return self.get_size()[0]


    @property
    def height(self):

        return self.get_size()[1]


class Monitor(threading.Thread):

    def __init__(self, collector_type):

        super(Monitor, self).__init__()
        self.daemon = True

        self.update_queue = multiprocessing.Queue(1)
        self.data_queue = multiprocessing.Queue(1)
        self.collector = collector_type(self.update_queue, self.data_queue)
        self.data = []

    def tick(self):
            
        #print "TICK"

        if not self.update_queue.full():
            #print "SEND UPDATE"
            self.update_queue.put('UPDATE', False)



    def start(self):

        self.collector.start()
        super(Monitor, self).start()


    def run(self):

        while True:
            data = self.data_queue.get(True)
            #print "GOT DATA!", data
            self.data = data


class Collector(multiprocessing.Process):

    def __init__(self, update_queue, data_queue):

        super(Collector, self).__init__()
        self.daemon = True
        self.update_queue = update_queue
        self.data_queue = data_queue


    def run(self):

        while True:

            msg = self.update_queue.get(True)
            if msg == 'UPDATE':
                #print "RECV UPDATE"
                self.update()


    def update(self):

        pass


class CPUCollector(Collector):

    def update(self):

        aggregate = psutil.cpu_percent(percpu=False, interval=0.5)
        percpu = psutil.cpu_percent(percpu=True, interval=0.5)

        self.data_queue.put([aggregate, percpu])



class Hugin(object):

    window = None
    monitors = None


    def __init__(self):

        self.window = Window()
        self.monitors = []
        self.monitors.append(Monitor(CPUCollector))
    
    def tick(self):

        for monitor in self.monitors:
            monitor.tick()
        self.window.queue_draw()


    def start(self):

        for monitor in self.monitors:
            monitor.start()

        t = PeriodicCall(self.tick, 0.03)
        t.start()
        
        signal.signal(signal.SIGINT, Gtk.main_quit) # so ctrl+c actually kills hugin
        Gtk.main()


hugin = Hugin()
hugin.start()
