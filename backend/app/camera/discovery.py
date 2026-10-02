from __future__ import annotations
"""Bounded ONVIF discovery on directly attached IPv4 networks; no port sweep."""
import base64
import hashlib
import ipaddress
import os
import select
import socket
import ssl
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, quote
from urllib.request import Request, build_opener, ProxyHandler, HTTPDigestAuthHandler, HTTPPasswordMgrWithDefaultRealm, HTTPRedirectHandler, HTTPSHandler
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape
from uuid import uuid4

import psutil

DEVICE = 'http://www.onvif.org/ver10/device/wsdl'
MEDIA = 'http://www.onvif.org/ver10/media/wsdl'


def xml(data):
    if len(data) > 1024 * 1024 or b'<!DOCTYPE' in data.upper() or b'<!ENTITY' in data.upper():
        raise ValueError('Invalid camera response')
    return ET.fromstring(data)


def same_host_url(url, host, schemes=('http', 'https')):
    parsed = urlsplit(url)
    if parsed.scheme not in schemes or parsed.hostname != host or parsed.username or parsed.password:
        raise ValueError('Camera returned an unexpected endpoint')
    return url


def parse_probe(data, host):
    if not ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback:
        return []
    root = xml(data)
    devices = []
    for match in root.findall('.//{*}ProbeMatch'):
        endpoint = match.findtext('.//{*}Address', '')
        for address in match.findtext('{*}XAddrs', '').split():
            try:
                same_host_url(address, host)
            except ValueError:
                continue
            devices.append({'id': 'onvif-' + hashlib.sha256((endpoint or host).encode()).hexdigest()[:20],
                            'name': f'Camera {host}', 'host': host, 'endpoint': address})
            break
    return devices


def probe(stop, seconds=4):
    sockets, devices = [], {}
    interfaces = []
    for addresses in psutil.net_if_addrs().values():
        for address in addresses:
            if address.family == socket.AF_INET:
                ip = ipaddress.ip_address(address.address)
                if ip.is_private and not ip.is_loopback:
                    interfaces.append(str(ip))
    try:
        for address in sorted(set(interfaces)):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind((address, 0))
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(address))
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
                message = f'''<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing" xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" xmlns:dn="http://www.onvif.org/ver10/network/wsdl"><s:Header><a:MessageID>uuid:{uuid4()}</a:MessageID><a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To><a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</a:Action></s:Header><s:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></s:Body></s:Envelope>'''
                sock.sendto(message.encode(), ('239.255.255.250', 3702))
                sockets.append(sock)
            except OSError:
                sock.close()
        deadline = time.monotonic() + seconds
        while sockets and time.monotonic() < deadline and not stop.is_set():
            for sock in select.select(sockets, [], [], .2)[0]:
                data, peer = sock.recvfrom(65535)
                try:
                    for device in parse_probe(data, peer[0]):
                        if len(devices) < 32:
                            devices[device['id']] = device
                except (ValueError, ET.ParseError):
                    continue
        return list(devices.values())
    finally:
        for sock in sockets:
            sock.close()


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def stream_uri(device, username='', password=''):
    from datetime import timedelta
    host, endpoint = device['host'], device['endpoint']
    passwords = HTTPPasswordMgrWithDefaultRealm()
    passwords.add_password(None, endpoint, username, password)
    ctx = ssl._create_unverified_context()
    opener = build_opener(ProxyHandler({}), NoRedirect(), HTTPDigestAuthHandler(passwords), HTTPSHandler(context=ctx))

    time_offset = [0.0]

    def soap(url, namespace, operation, content='', allow_retry=True):
        same_host_url(url, host)
        passwords.add_password(None, url, username, password)
        security = ''
        if username:
            nonce = os.urandom(16)
            created_dt = datetime.now(timezone.utc) + timedelta(seconds=time_offset[0])
            created = created_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            digest = base64.b64encode(hashlib.sha1(nonce + created.encode() + password.encode()).digest()).decode()
            security = f'''<wsse:Security xmlns:wsse="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd" xmlns:wsu="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"><wsse:UsernameToken><wsse:Username>{escape(username)}</wsse:Username><wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">{digest}</wsse:Password><wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">{base64.b64encode(nonce).decode()}</wsse:Nonce><wsu:Created>{created}</wsu:Created></wsse:UsernameToken></wsse:Security>'''
        body = f'<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:m="{namespace}" xmlns:trt="{MEDIA}" xmlns:tt="http://www.onvif.org/ver10/schema"><s:Header>{security}</s:Header><s:Body><m:{operation}>{content}</m:{operation}></s:Body></s:Envelope>'
        req = Request(url, data=body.encode(), headers={'Content-Type': f'application/soap+xml; charset=utf-8; action="{namespace}/{operation}"'})
        try:
            with opener.open(req, timeout=3) as response:
                root = xml(response.read(1024 * 1024 + 1))
        except Exception as exc:
            err_body = getattr(exc, 'read', lambda: b'')() if hasattr(exc, 'read') else b''
            if allow_retry and (b'time check failed' in err_body.lower() or b'notauthorized' in err_body.lower()):
                try:
                    time_body = '<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope" xmlns:d="http://www.onvif.org/ver10/device/wsdl"><s:Body><d:GetSystemDateAndTime/></s:Body></s:Envelope>'
                    time_req = Request(endpoint, data=time_body.encode(), headers={'Content-Type': 'application/soap+xml; charset=utf-8; action="http://www.onvif.org/ver10/device/wsdl/GetSystemDateAndTime"'})
                    with opener.open(time_req, timeout=3) as r:
                        t_root = xml(r.read(1024 * 1024 + 1))
                        year = int(t_root.findtext('.//{*}Date/{*}Year') or 0)
                        month = int(t_root.findtext('.//{*}Date/{*}Month') or 0)
                        day = int(t_root.findtext('.//{*}Date/{*}Day') or 0)
                        hour = int(t_root.findtext('.//{*}Time/{*}Hour') or 0)
                        minute = int(t_root.findtext('.//{*}Time/{*}Minute') or 0)
                        second = int(t_root.findtext('.//{*}Time/{*}Second') or 0)
                        if year > 2000:
                            cam_dt = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                            time_offset[0] = (cam_dt - datetime.now(timezone.utc)).total_seconds()
                    return soap(url, namespace, operation, content, allow_retry=False)
                except Exception:
                    pass
            raise
        if root.find('.//{*}Fault') is not None:
            raise ValueError('Camera service rejected the request')
        return root

    capabilities = soap(endpoint, DEVICE, 'GetCapabilities', '<m:Category>Media</m:Category>')
    media = capabilities.findtext('.//{*}Media/{*}XAddr', '')
    same_host_url(media, host)
    profiles = soap(media, MEDIA, 'GetProfiles')
    profile = profiles.find('.//{*}Profiles')
    if profile is None or not profile.get('token'):
        raise ValueError('No camera media profile')
    result = soap(media, MEDIA, 'GetStreamUri', '<trt:StreamSetup><tt:Stream>RTP-Unicast</tt:Stream><tt:Transport><tt:Protocol>RTSP</tt:Protocol></tt:Transport></trt:StreamSetup><trt:ProfileToken>' + escape(profile.get('token')) + '</trt:ProfileToken>')
    uri = result.findtext('.//{*}Uri', '')
    same_host_url(uri, host, ('rtsp', 'rtsps'))
    parsed = urlsplit(uri)
    authority = parsed.netloc
    if username:
        authority = quote(username, safe='') + ':' + quote(password, safe='') + '@' + authority
    scheme = parsed.scheme
    if endpoint.startswith('https') and scheme == 'rtsp':
        scheme = 'rtsps'
    return urlunsplit((scheme, authority, parsed.path, parsed.query, ''))
