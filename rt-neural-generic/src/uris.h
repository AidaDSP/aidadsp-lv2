/*
  This file taken from the LV2 ImpulseResponser Example Plugin
  Copyright 2011-2012 David Robillard <d@drobilla.net>

  Permission to use, copy, modify, and/or distribute this software for any
  purpose with or without fee is hereby granted, provided that the above
  copyright notice and this permission notice appear in all copies.

  THIS SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

  Modified August 2022 by Massimo Pennazio maxipenna@libero.it
*/
#pragma once

#include <lv2/log/log.h>
#include <lv2/midi/midi.h>
#include <lv2/state/state.h>
#include <lv2/parameters/parameters.h>

#ifndef PLUGIN_URI
#define PLUGIN_URI "http://aidadsp.cc/plugins/aidadsp-bundle/rt-neural-generic"
#endif

#define PLUGIN__json PLUGIN_URI "#json"
#define PLUGIN__applyJson PLUGIN_URI "#applyJson"

typedef struct {
    LV2_URID atom_Float;
    LV2_URID atom_Path;
    LV2_URID atom_Resource;
    LV2_URID atom_Sequence;
    LV2_URID atom_URID;
    LV2_URID atom_eventTransfer;
    LV2_URID applyJson;
    LV2_URID json;
    LV2_URID midi_Event;
    LV2_URID param_gain;
    LV2_URID patch_Get;
    LV2_URID patch_Set;
    LV2_URID patch_property;
    LV2_URID patch_value;
} PluginURIs;

static inline void
map_plugin_uris(LV2_URID_Map* map, PluginURIs* uris)
{
    uris->atom_Float               = map->map(map->handle, LV2_ATOM__Float);
    uris->atom_Path                = map->map(map->handle, LV2_ATOM__Path);
    uris->atom_Resource            = map->map(map->handle, LV2_ATOM__Resource);
    uris->atom_Sequence            = map->map(map->handle, LV2_ATOM__Sequence);
    uris->atom_URID                = map->map(map->handle, LV2_ATOM__URID);
    uris->atom_eventTransfer       = map->map(map->handle, LV2_ATOM__eventTransfer);
    uris->applyJson                = map->map(map->handle, PLUGIN__applyJson);
    uris->json                     = map->map(map->handle, PLUGIN__json);
    uris->midi_Event               = map->map(map->handle, LV2_MIDI__MidiEvent);
    uris->param_gain               = map->map(map->handle, LV2_PARAMETERS__gain);
    uris->patch_Get                = map->map(map->handle, LV2_PATCH__Get);
    uris->patch_Set                = map->map(map->handle, LV2_PATCH__Set);
    uris->patch_property           = map->map(map->handle, LV2_PATCH__property);
    uris->patch_value              = map->map(map->handle, LV2_PATCH__value);
}

/**
 * Write a message like the following to @p forge:
 * []
 *     a patch:Set ;
 *     patch:property eg:json ;
 *     patch:value </home/me/foo.json> .
 */
static inline LV2_Atom*
write_set_file(LV2_Atom_Forge*    forge,
               const PluginURIs* uris,
               const char*        filename,
               const uint32_t     filename_len)
{
    LV2_Atom_Forge_Frame frame;
    LV2_Atom* set = (LV2_Atom*)lv2_atom_forge_object(
                forge, &frame, 0, uris->patch_Set);

    lv2_atom_forge_key(forge, uris->patch_property);
    lv2_atom_forge_urid(forge, uris->json);
    lv2_atom_forge_key(forge, uris->patch_value);
    lv2_atom_forge_path(forge, filename, filename_len + 1);

    lv2_atom_forge_pop(forge, &frame);

    return set;
}
