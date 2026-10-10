# Home network

The router hands out addresses from a private /24 range and forwards DNS to a local resolver that blocks ads.
The file server keeps a fixed lease so that backups and the media player can find it.

## Wi-Fi drops on the laptop

When the laptop loses Wi-Fi after waking from sleep, disable power saving on the wireless adapter and update
its driver. The access point logs showed the client never sent a disassociation frame, which points at the
laptop rather than the router.

Unrelated, but written down on the same evening: the bread recipe is in [[sourdough]].
