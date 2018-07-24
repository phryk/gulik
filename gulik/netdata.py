#!/usr/bin/env python3.6

""" a simple module for querying netdata REST APIs """

import time
import requests


class NetdataException(Exception):
    pass


class NetdataWarning(NetdataException):
    pass


class NetdataError(NetdataException):
    pass


class Netdata(object):

    def __init__(self, host, port=None):

        self.host = host
        self.port = port or 19999

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}/api/v1"


    def _get(self, path, params):

        url = f"{self.base_url}/{path}"
        killkeys = []
        for k, v in params.items():
            if v is None:
                killkeys.append(k)

        for k in killkeys:
                del(params[k])
        
        try:
            response = requests.get(url, params=params)
            #print(response.url)
            if response.status_code == 200:
                try:

                    return response.json()
                except ValueError as e:
                    raise NetdataError(f"Could not decode response from {url} as JSON: {e}")
            else:
                raise NetdataError(f"Non-200 status code for {response.url}")

        except requests.exceptions.ConnectionError as e:
            raise NetdataError(f"Could not connect to netdata instance {self.host}:{self.port}!")

        except Exception as e:
            raise NetdataWarning(f"Got exception while trying to communicate with netdata instance {self.host}:{self.port}: {e}")


    def charts(self):

        return self._get('charts', {})


    def chart(self, chart=None):

        return self._get('chart', {'chart': chart})

    
    def data(self, chart=None, dimension=None, before=None, after=None, points=None, group=None, gtime=None, options=None, callback=None, filename=None, tqx=None):

        """ doesn't offer the format parameter because I'm too lazy. """

        return self._get('data', {
                'chart': chart,
                'dimension': dimension,
                'before': before,
                'after': after,
                'points': points,
                'group': group,
                'gtime': gtime,
                'options': options,
                'callback': callback,
                'filename': filename,
                'tqx': tqx
            }
        )


    def allmetrics(self, help=None, types=None, timestamps=None, names=None, server=None, prefix=None, data=None):

        return self._get('allmetrics', {
                'help': 'yes' if help else 'no',
                'types': types,
                'timestamps': timestamps,
                'names': names,
                'server': server,
                'prefix': prefix,
                'data': data
            }
        )


# Usage:
#c = Netdata('localhost')
#c = Netdata('frankfurt.my-netdata.io', port=80)
#c = Netdata('ventureer.my-netdata.io', port=80)
#
#for name in c.charts()['charts'].keys():
    #print(c.chart(name).keys())
#    print(c.data(name, points=1, after=-1)['data'])
#
#while True:
#    resp = c.data('system.ram', points=1, options=['unaligned'])
#    #resp = c.data('system.ram', points=1, after=-1)
#    print(len(resp['data']), resp['data'])
#    time.sleep(0.3)
