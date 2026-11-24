"""Native Windows startup under our process-tree supervisor."""
import os


def configure_supervised_windows_ipc():
    if os.name != 'nt' or os.environ.get('CS557_SUPERVISED_WINDOWS') != '1':
        return False
    from craftground.environment.socket_ipc import SocketIPC
    # CraftGround's routine scans open files for Unix /tmp/minecraftrl_ sockets
    # across every process. Native Windows uses TCP, and open_files() can stall
    # on unrelated handles. windows_run owns/cleans this job's descendants;
    # queue workers separately require an idle Minecraft runtime and TCP port.
    SocketIPC.remove_orphan_java_processes = lambda self: None
    return True
