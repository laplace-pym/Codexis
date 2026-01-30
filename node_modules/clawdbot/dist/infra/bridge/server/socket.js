export function configureNodeBridgeSocket(socket) {
    socket.setNoDelay(true);
    socket.setKeepAlive(true, 15_000);
}
