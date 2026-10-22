/**
 * @file     menu_stubs.c
 * @brief    NUCLEO-WBA55CG: the TMAP application updates the DK1 screen menu at every state
 *           change; without a screen these calls do nothing. Connection and stream events are
 *           still logged on the ST-LINK virtual COM port by the application itself.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "app_menu_cfg.h"

void Menu_Config(void) {}
void Menu_SetConnectingPage(void) {}
void Menu_SetNoStreamPage(void) {}
void Menu_SetStreamingPage(char *pSamplerateText, Audio_Role_t AudioRole) { (void)pSamplerateText; (void)AudioRole; }
void Menu_SetWaitConnPage(void) {}
void Menu_SetStartupPage(void) {}
void Menu_SetIdentifier(char *pId, uint8_t len) { (void)pId; (void)len; }
void Menu_SetVolume(uint8_t Volume) { (void)Volume; }
void Menu_SetCallState(char *pCallState) { (void)pCallState; }
void Menu_SetMediaState(char *pMediaState) { (void)pMediaState; }
void Menu_SetTrackTitle(uint8_t *pTrackTitle, uint8_t TrackNameLen) { (void)pTrackTitle; (void)TrackNameLen; }
void Menu_AddBroadcastSource(uint8_t AdvSID, uint8_t *pAddress, uint8_t AddressType, char *pDeviceName)
{
  (void)AdvSID;
  (void)pAddress;
  (void)AddressType;
  (void)pDeviceName;
}
void Menu_SetPASyncPage(void) {}
void Menu_SetBISSyncPage(void) {}
void Menu_SetBroadcastSyncedPage(char *pSamplerateText) { (void)pSamplerateText; }
void Menu_SetBroadcastScanPage(void) {}
void Menu_SetScanDelegSyncedPage(char *pSamplerateText) { (void)pSamplerateText; }
void Menu_SetProfilesLinked(audio_profile_t Profiles) { (void)Profiles; }
