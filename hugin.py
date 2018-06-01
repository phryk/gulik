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

    daemon = True
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



        self.show_all()
        self.move(0, 32) # move apparently must be called after show_all


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

        self.queue_update = multiprocessing.Queue(1)
        self.queue_data = multiprocessing.Queue(1)
        self.collector = collector_type(self.queue_update, self.queue_data)
        self.data = []


    def tick(self):
            
        print "MONITOR TICK START"

        if not self.queue_update.full():
            print "SEND UPDATE"
            self.queue_update.put('UPDATE', False)

        print "MONITOR TICK END"



    def start(self):

        self.collector.start()
        super(Monitor, self).start()


    def run(self):

        while self.collector.is_alive():
            data = self.queue_data.get(True)
            #print "GOT DATA!", data
            self.data = data


class Collector(multiprocessing.Process):

    def __init__(self, queue_update, queue_data):

        super(Collector, self).__init__()
        self.daemon = True
        self.queue_update = queue_update
        self.queue_data = queue_data


    def run(self):

        while True:

            msg = self.queue_update.get(True)
            if msg == 'UPDATE':
                #print "RECV UPDATE"
                self.update()


    def update(self):

        pass


class CPUCollector(Collector):

    def update(self):

        aggregate = psutil.cpu_percent(percpu=False)#, interval=0.5)
        percpu = psutil.cpu_percent(percpu=True)#, interval=0.5)

        self.queue_data.put([aggregate, percpu])

        time.sleep(0.1) # according to psutil docs, there should at least be 0.1 seconds between calling cpu_percent without interval




class Hugin(object):

    window = None
    monitors = None


    def __init__(self):

        self.window = Window()
        self.window.connect('draw', self.draw)
        
        self.monitors = {}
        self.monitors['cpu'] = Monitor(CPUCollector)


    def tick(self):

        for monitor in self.monitors.itervalues():
            monitor.tick()
        self.window.queue_draw()


    def draw(self, window, context):

        context.set_operator(cairo.OPERATOR_CLEAR)
        context.paint()
        context.set_operator(cairo.OPERATOR_OVER)

        context.set_source_rgba(0.5, 1, 0, 0.2)
        context.rectangle(0, 0, random.randint(0, self.window.width), random.randint(0, self.window.height))
        context.fill()
        print "draw"
        if len(self.monitors['cpu'].data):
            context.set_source_rgba(0.5, 1, 0, 0.6)
            current_cpu = self.monitors['cpu'].data[0]
            context.rectangle(0, 0, 20, current_cpu * 3)
            print current_cpu 
            context.fill()


    def start(self):

        for monitor in self.monitors.itervalues():
            monitor.start()

        t = PeriodicCall(self.tick, 0.016)
        t.start()
        
        signal.signal(signal.SIGINT, Gtk.main_quit) # so ctrl+c actually kills hugin
        Gtk.main()


hugin = Hugin()
hugin.start()
