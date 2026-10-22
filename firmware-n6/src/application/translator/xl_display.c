/**
 * @file     xl_display.c
 * @brief    Translator screen and power states; implements xl_display.h.
 * @details  Power states:
 *           - TRANSLATING: the N6 has the speaker; recognition runs, the speaker stream runs, the
 *             screen is on.
 *           - MUSIC_SCREEN: the WBA has the speaker; recognition is paused and the speaker stream
 *             stopped, the screen stays on for a few seconds (after the switch, or after a touch).
 *           - MUSIC_SLEEP: as MUSIC_SCREEN with the panel and its backlight off; the inputs are
 *             polled 5 times a second.
 *           The state follows the router's target source, so a switch in progress already counts.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "xl_display.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "asr_model_params.h"
#include "audio_port.h"
#include "logger.h"
#include "mw_lcd.h"
#include "stm32n6570_discovery.h"
#include "stm32n6570_discovery_bus.h"
#include "stm32n6570_discovery_ts.h"
#include "stm32n6570dk.h"
#include "text_case.h"
#include "xl_audio_out.h"
#include "xl_router.h"

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

/** @brief Poll period while the screen is on, and while asleep. */
#define XL_POLL_AWAKE_MS  40U
#define XL_POLL_ASLEEP_MS 200U

/** @brief The start-up logo stays this long; the touch screen and the speaker start after it. */
#define XL_LOGO_MS 1500U

/** @brief Touch controller probes after the logo (it shares the LCD reset). */
#define XL_TOUCH_ATTEMPTS 5U
#define XL_TOUCH_RETRY_MS 200U

/** @brief Taps closer than this count once. */
#define XL_PRESS_LOCKOUT_MS 250U

/** @brief Screen on after music takes the speaker, and after a touch while music plays. */
#define XL_MUSIC_SCREEN_MS 5000U
#define XL_WAKE_SCREEN_MS  10000U

/** @brief A note (e.g. "tone") replaces the Hindi line this long. */
#define XL_NOTE_MS 3000U

/** @brief Recognized English kept for the screen, and the tentative tail. */
#define XL_TEXT_CAPACITY      400U
#define XL_TENTATIVE_CAPACITY 160U

/** @brief Layout, in pixels (800 x 480 panel, 48 px font). */
#define XL_MARGIN     12
#define XL_BAR_H      64
#define XL_BUTTON_H   84
#define XL_BUTTON_GAP 10

/** @brief TONE button test tone: about 0.08 W into the 8 Ohm speaker. */
#define XL_TONE_HZ        880U
#define XL_TONE_MS        700U
#define XL_TONE_AMPLITUDE 20000

/** @brief Buttons, left to right. */
enum
{
    BUTTON_AUTO = 0,
    BUTTON_TRANSLATE,
    BUTTON_MUSIC,
    BUTTON_TONE,
    BUTTON_COUNT
};

/*******************************************************************************/
/*                                 PRIVATE TYPES                               */
/*******************************************************************************/

/** @brief Power states, see the file description. */
typedef enum
{
    STATE_TRANSLATING = 0,
    STATE_MUSIC_SCREEN,
    STATE_MUSIC_SLEEP
} power_state_t;

/** @brief A touch button. */
typedef struct
{
    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;
    const char *label;
} button_t;

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

static button_t buttons[BUTTON_COUNT] = {
    { 0, 0, 0, 0, "AUTO" },
    { 0, 0, 0, 0, "TRANSLATE" },
    { 0, 0, 0, 0, "MUSIC" },
    { 0, 0, 0, 0, "TONE" },
};

/** @brief Recognized English, newest last, NUL-terminated. */
static char english[XL_TEXT_CAPACITY + 1U];
static uint32_t english_length;

/** @brief Newest tentative tail, NUL-terminated. */
static char tentative[XL_TENTATIVE_CAPACITY];

/** @brief English as drawn: final text and tentative tail, in sentence case for punctuation models. */
static char shown[XL_TEXT_CAPACITY + XL_TENTATIVE_CAPACITY + 1U];

static power_state_t state = STATE_TRANSLATING;
static uint32_t screen_until;

/** @brief Bumped when something on screen changed; the screen is redrawn when it differs. */
static uint32_t revision = 1U;
static uint32_t drawn_revision;

static uint32_t started_at;
static int started;
static int laid_out;
static int audio_ok;
static int chime_pending;

static int touch_ready;
static int touch_down;
static int button_down;
static uint32_t last_press;

static const char *note;
static uint32_t note_until;

static xl_mode_t drawn_mode;
static int drawn_music;
static xl_source_t drawn_source;

static uint16_t color_blue;
static uint16_t color_green;
static uint16_t color_orange;
static uint16_t color_grey;
static uint16_t color_light;
static uint16_t color_dark;

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/** @brief Milliseconds since start-up. */
static uint32_t now_ms(void)
{
    return (uint32_t)(((uint64_t)tx_time_get() * 1000U) / TX_TIMER_TICKS_PER_SECOND);
}

/** @brief Panel and backlight on or off; the LTDC keeps scanning the framebuffer. */
static void lcd_power(int on)
{
    HAL_GPIO_WritePin(BOARD_LCD_CTRL_PORT, BOARD_LCD_CTRL_PINS, on ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

/** @brief Show @p text instead of the Hindi line for a while. */
static void show_note(const char *text, uint32_t now)
{
    note = text;
    note_until = now + XL_NOTE_MS;
    ++revision;
}

/** @brief Append committed English, dropping the oldest words when full. */
static void append_english(const char *text, uint32_t length)
{
    if (length > XL_TEXT_CAPACITY)
    {
        text += length - XL_TEXT_CAPACITY;
        length = XL_TEXT_CAPACITY;
    }
    if ((english_length + length) > XL_TEXT_CAPACITY)
    {
        uint32_t drop = (english_length + length) - XL_TEXT_CAPACITY;

        while ((drop < english_length) && (english[drop] != ' '))
        {
            ++drop;
        }
        if (drop > english_length)
        {
            drop = english_length;
        }
        (void)memmove(english, &english[drop], english_length - drop);
        english_length -= drop;
    }
    (void)memcpy(&english[english_length], text, length);
    english_length += length;
    english[english_length] = '\0';
}

/** @brief Advance of one character of the 48 px font. */
static int32_t char_width(char character)
{
    const char text[2] = { character, '\0' };

    return (int32_t)mw_lcd_measure_string_width(text, 1U);
}

/**
 * @brief Word-wrap @p text into a box and optionally draw it.
 * @param[in] text Text to lay out.
 * @param[in] grey_from Index from which the characters are tentative (drawn grey).
 * @param[in] x0 Left edge of the box.
 * @param[in] y0 Top edge of the box.
 * @param[in] width Width of the box.
 * @param[in] skip Lines laid out but not drawn (the oldest).
 * @param[in] draw Zero to only count the lines.
 * @return Lines the text occupies.
 */
static uint32_t wrap_text(const char *text, uint32_t grey_from, int32_t x0, int32_t y0, int32_t width,
                          uint32_t skip, int draw)
{
    const int32_t line_height = (int32_t)mw_lcd_get_line_height(1U);
    const uint32_t total = (uint32_t)strlen(text);
    int32_t x = 0;
    uint32_t line = 0U;
    uint32_t index = 0U;

    while (index < total)
    {
        uint32_t end = index;
        int32_t word = 0;

        if (text[index] == ' ')
        {
            if (x > 0)
            {
                x += char_width(' ');
            }
            ++index;
            continue;
        }
        while ((end < total) && (text[end] != ' '))
        {
            word += char_width(text[end]);
            ++end;
        }
        if ((x > 0) && ((x + word) > width))
        {
            x = 0;
            ++line;
        }
        for (; index < end; ++index)
        {
            const int32_t advance = char_width(text[index]);

            if ((x > 0) && ((x + advance) > width))
            {
                x = 0;
                ++line;
            }
            if (draw && (line >= skip))
            {
                mw_lcd_draw_char((uint16_t)(x0 + x), (uint16_t)(y0 + ((int32_t)(line - skip) * line_height)),
                                 text[index], 1U, (index >= grey_from) ? color_grey : MW_LCD_COLOR_BLACK, true);
            }
            x += advance;
        }
    }
    return (total > 0U) ? (line + 1U) : 0U;
}

/** @brief Draw @p text so its last lines fill a box of @p max_lines lines. */
static void draw_text_box(const char *text, uint32_t grey_from, int32_t x0, int32_t y0, int32_t width,
                          uint32_t max_lines)
{
    const uint32_t lines = wrap_text(text, grey_from, x0, y0, width, 0U, 0);
    const uint32_t skip = (lines > max_lines) ? (lines - max_lines) : 0U;

    (void)wrap_text(text, grey_from, x0, y0, width, skip, 1);
}

/** @brief Draw @p text centered horizontally in [x, x + w) at @p y. */
static void draw_centered(int32_t x, int32_t w, int32_t y, const char *text, uint16_t color)
{
    const int32_t width = (int32_t)mw_lcd_measure_string_width(text, 1U);
    const int32_t left = x + ((w - width) / 2);

    mw_lcd_draw_string((uint16_t)((left > 0) ? left : 0), (uint16_t)y, text, 1U, color, true);
}

/** @brief Size the buttons to their labels and spread them over the bottom row. */
static void layout_buttons(void)
{
    const int32_t screen_w = (int32_t)mw_lcd_get_width();
    const int32_t y = (int32_t)mw_lcd_get_height() - XL_MARGIN - XL_BUTTON_H;
    int32_t label_total = 0;
    int32_t spare;
    int32_t x = XL_MARGIN;
    uint32_t index;

    for (index = 0U; index < BUTTON_COUNT; ++index)
    {
        buttons[index].w = (int32_t)mw_lcd_measure_string_width(buttons[index].label, 1U);
        label_total += buttons[index].w;
    }
    /* What is left after the labels, the margins and the gaps is shared as padding. */
    spare = screen_w - (2 * XL_MARGIN) - ((int32_t)(BUTTON_COUNT - 1U) * XL_BUTTON_GAP) - label_total;
    for (index = 0U; index < BUTTON_COUNT; ++index)
    {
        buttons[index].w += (spare > 0) ? (spare / (int32_t)BUTTON_COUNT) : 0;
        buttons[index].x = x;
        buttons[index].y = y;
        buttons[index].h = XL_BUTTON_H;
        x += buttons[index].w + XL_BUTTON_GAP;
    }
    laid_out = 1;
}

/** @brief Is button @p index the one for the current mode? */
static int button_selected(uint32_t index)
{
    const xl_mode_t mode = xl_router_mode();

    return ((index == BUTTON_AUTO) && (mode == XL_MODE_AUTO)) ||
           ((index == BUTTON_TRANSLATE) && (mode == XL_MODE_TRANSLATE)) ||
           ((index == BUTTON_MUSIC) && (mode == XL_MODE_MUSIC));
}

/** @brief The status line: who has the speaker, and whether music is waiting. */
static const char *status_text(void)
{
    const int music = xl_router_music();

    if (xl_router_target() == XL_SOURCE_N6)
    {
        return music ? "Translator (music waiting)" : "Speaker: Translator";
    }
    return music ? "Speaker: Bluetooth music" : "Music (nothing playing)";
}

/** @brief Draw the whole screen and present it. */
static void draw_screen(uint32_t now)
{
    const int32_t screen_w = (int32_t)mw_lcd_get_width();
    const int32_t line_height = (int32_t)mw_lcd_get_line_height(1U);
    const int32_t text_w = screen_w - (2 * XL_MARGIN);
    const int32_t english_y = XL_BAR_H + 8;
    const int32_t hindi_y = english_y + (3 * line_height) + 16;
    uint32_t index;

    if (!laid_out)
    {
        layout_buttons();
    }
    mw_lcd_clear(MW_LCD_COLOR_WHITE);

    /* Status bar. */
    mw_lcd_fill_rect(0, 0, screen_w, XL_BAR_H, (xl_router_target() == XL_SOURCE_N6) ? color_blue : color_green);
    draw_centered(0, screen_w, (XL_BAR_H - line_height) / 2, status_text(), MW_LCD_COLOR_WHITE);

    if (state == STATE_TRANSLATING)
    {
        /* English: final text black, tentative tail grey, newest lines at the bottom of 3. */
        const uint32_t final_length = english_length;

        (void)memcpy(shown, english, final_length);
        (void)strcpy(&shown[final_length], tentative);
#if defined(ASR_MODEL_NET_PUNCT_FIRST_INDEX)
        text_sentence_case(shown, (uint32_t)strlen(shown), 1);
#endif
        if (shown[0] == '\0')
        {
            mw_lcd_draw_string((uint16_t)XL_MARGIN, (uint16_t)english_y, "Speak English...", 1U, color_grey, true);
        }
        else
        {
            draw_text_box(shown, final_length, XL_MARGIN, english_y, text_w, 3U);
        }
    }
    else
    {
        mw_lcd_draw_string((uint16_t)XL_MARGIN, (uint16_t)english_y, "Music from Bluetooth.", 1U, color_dark, true);
        mw_lcd_draw_string((uint16_t)XL_MARGIN, (uint16_t)(english_y + line_height),
                           "Translator asleep.", 1U, color_grey, true);
        mw_lcd_draw_string((uint16_t)XL_MARGIN, (uint16_t)(english_y + (2 * line_height)),
                           "Tap TRANSLATE to talk.", 1U, color_grey, true);
    }

    /* Hindi (or a note). */
    mw_lcd_draw_line(XL_MARGIN, hindi_y - 8, screen_w - XL_MARGIN, hindi_y - 8, 2U, color_light);
    if ((note != NULL) && ((int32_t)(note_until - now) > 0))
    {
        draw_text_box(note, 0xFFFFFFFFU, XL_MARGIN, hindi_y, text_w, 2U);
    }
    else
    {
        note = NULL;
        mw_lcd_draw_string((uint16_t)XL_MARGIN, (uint16_t)hindi_y, "Hindi: (translation model", 1U, color_grey, true);
        mw_lcd_draw_string((uint16_t)XL_MARGIN, (uint16_t)(hindi_y + line_height), "not installed yet)", 1U,
                           color_grey, true);
    }

    /* Buttons. */
    for (index = 0U; index < BUTTON_COUNT; ++index)
    {
        const button_t *b = &buttons[index];
        const int selected = button_selected(index);
        const uint16_t fill = (index == BUTTON_TONE) ? color_orange : (selected ? color_blue : color_light);
        const uint16_t ink = ((index == BUTTON_TONE) || selected) ? MW_LCD_COLOR_WHITE : color_dark;

        mw_lcd_fill_rect(b->x, b->y, b->w, b->h, fill);
        draw_centered(b->x, b->w, b->y + ((b->h - line_height) / 2), b->label, ink);
    }

    if (mw_lcd_present() != MW_LCD_OK)
    {
        LOG_ERROR("Translator: LCD present failed");
    }
    drawn_revision = revision;
    drawn_mode = xl_router_mode();
    drawn_music = xl_router_music();
    drawn_source = xl_router_target();
}

/** @brief Bring up the touch screen with retries (it boots with the LCD, after the reset). */
static void touch_start(void)
{
    TS_Init_t ts;
    int32_t status = BSP_ERROR_NO_INIT;
    uint32_t attempt;

    ts.Width = mw_lcd_get_width();
    ts.Height = mw_lcd_get_height();
    ts.Orientation = TS_SWAP_NONE;
    ts.Accuracy = 0U;
    for (attempt = 1U; attempt <= XL_TOUCH_ATTEMPTS; ++attempt)
    {
        status = BSP_TS_Init(0U, &ts);
        if (status == BSP_ERROR_NONE)
        {
            break;
        }
        (void)tx_thread_sleep((XL_TOUCH_RETRY_MS * TX_TIMER_TICKS_PER_SECOND) / 1000U);
    }
    touch_ready = (status == BSP_ERROR_NONE);
    if (touch_ready)
    {
        LOG_INFO("Translator: touch screen ready (attempt %lu)", (unsigned long)attempt);
    }
    else
    {
        LOG_WARNING("Translator: touch screen not found (BSP_TS_Init %ld); use the USER button", (long)status);
    }
}

/**
 * @brief A new tap or USER-button press.
 * @param[in] now Milliseconds since start-up.
 * @param[out] x Tap position; -1 for the USER button.
 * @param[out] y Tap position; -1 for the USER button.
 * @return 1 on a new press.
 */
static int read_press(uint32_t now, int32_t *x, int32_t *y)
{
    int press = 0;
    const int button = (BSP_PB_GetState(BUTTON_USER1) != 0U);

    *x = -1;
    *y = -1;
    if (button && !button_down)
    {
        press = 1;
    }
    button_down = button;
    if (touch_ready)
    {
        TS_State_t ts;

        if (BSP_TS_GetState(0U, &ts) == BSP_ERROR_NONE)
        {
            const int down = (ts.TouchDetected != 0U);

            if (down && !touch_down)
            {
                press = 1;
                *x = (int32_t)ts.TouchX;
                *y = (int32_t)ts.TouchY;
            }
            touch_down = down;
        }
    }
    if (press && ((now - last_press) < XL_PRESS_LOCKOUT_MS))
    {
        press = 0;
    }
    if (press)
    {
        last_press = now;
    }
    return press;
}

/** @brief The button under (x, y), or BUTTON_COUNT. */
static uint32_t hit_button(int32_t x, int32_t y)
{
    uint32_t index;

    for (index = 0U; index < BUTTON_COUNT; ++index)
    {
        const button_t *b = &buttons[index];

        if ((x >= (b->x - 6)) && (x < (b->x + b->w + 6)) && (y >= (b->y - 20)))
        {
            return index;
        }
    }
    return BUTTON_COUNT;
}

/** @brief Act on a tap (or the USER button when x is -1). */
static void handle_press(int32_t x, int32_t y, uint32_t now)
{
    uint32_t index;

    if (x < 0)
    {
        /* USER button: next mode, AUTO -> TRANSLATE -> MUSIC -> AUTO. */
        xl_router_set_mode((xl_mode_t)(((uint32_t)xl_router_mode() + 1U) % 3U));
        ++revision;
        return;
    }
    index = hit_button(x, y);
    switch (index)
    {
    case BUTTON_AUTO:
        xl_router_set_mode(XL_MODE_AUTO);
        break;
    case BUTTON_TRANSLATE:
        xl_router_set_mode(XL_MODE_TRANSLATE);
        break;
    case BUTTON_MUSIC:
        xl_router_set_mode(XL_MODE_MUSIC);
        break;
    case BUTTON_TONE:
        if ((xl_router_source() == XL_SOURCE_N6) && audio_ok && xl_audio_out_running())
        {
            (void)xl_audio_out_tone(XL_TONE_HZ, XL_TONE_MS, XL_TONE_AMPLITUDE);
            show_note("Test tone: 880 Hz on the speaker", now);
            LOG_INFO("Translator: test tone, %lu underruns so far", (unsigned long)xl_audio_out_underruns());
        }
        else
        {
            show_note("The speaker plays music now: tap TRANSLATE first", now);
        }
        break;
    default:
        break;
    }
    ++revision;
}

/** @brief The WBA takes the speaker: the N6 stops listening and speaking, the screen stays a while. */
static void enter_music(uint32_t now)
{
    LOG_INFO("Translator: music has the speaker, N6 going to sleep");
    audio_port_set_paused(1);
    if (audio_ok)
    {
        xl_audio_out_flush();
        xl_audio_out_stop();
    }
    state = STATE_MUSIC_SCREEN;
    screen_until = now + XL_MUSIC_SCREEN_MS;
    ++revision;
}

/** @brief Screen off; inputs still polled. */
static void enter_sleep(void)
{
    LOG_INFO("Translator: asleep (screen off)");
    lcd_power(0);
    state = STATE_MUSIC_SLEEP;
}

/** @brief A touch while asleep: screen on for a while, still asleep otherwise. */
static void wake_screen(uint32_t now)
{
    lcd_power(1);
    state = STATE_MUSIC_SCREEN;
    screen_until = now + XL_WAKE_SCREEN_MS;
    ++revision;
}

/** @brief The N6 has the speaker again: listen, speak, screen on. */
static void wake_translator(void)
{
    LOG_INFO("Translator: awake, listening");
    if (state == STATE_MUSIC_SLEEP)
    {
        lcd_power(1);
    }
    audio_port_set_paused(0);
    if (audio_ok && (xl_audio_out_start() != 0))
    {
        LOG_ERROR("Translator: speaker stream did not restart");
    }
    state = STATE_TRANSLATING;
    chime_pending = 1;
    ++revision;
}

/** @brief Start the speaker output once the microphone has set up the audio clock. */
static void start_speaker(void)
{
    const int status = xl_audio_out_init();

    if (status != 0)
    {
        LOG_ERROR("Translator: speaker output init failed (%d)", status);
        return;
    }
    if (xl_audio_out_start() != 0)
    {
        LOG_ERROR("Translator: speaker stream did not start");
        return;
    }
    audio_ok = 1;
    chime_pending = 1;
    LOG_INFO("Translator: speaker output on SAI2 (A2/D6/D8), switch on D7, amp SD on D4, music in on D2");
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

void xl_display_start(void)
{
    color_blue = mw_lcd_rgb888_to_rgb565(0x1EU, 0x6FU, 0xD9U);
    color_green = mw_lcd_rgb888_to_rgb565(0x1EU, 0x9EU, 0x3EU);
    color_orange = mw_lcd_rgb888_to_rgb565(0xE0U, 0x8AU, 0x00U);
    color_grey = mw_lcd_rgb888_to_rgb565(0x80U, 0x80U, 0x80U);
    color_light = mw_lcd_rgb888_to_rgb565(0xE0U, 0xE0U, 0xE0U);
    color_dark = mw_lcd_rgb888_to_rgb565(0x30U, 0x30U, 0x30U);

    if (BSP_PB_Init(BUTTON_USER1, BUTTON_MODE_GPIO) != BSP_ERROR_NONE)
    {
        LOG_WARNING("Translator: USER button not available");
    }
    xl_router_init();
    started_at = now_ms();
}

void xl_display_text(const char *final_text, uint32_t final_length, const char *tentative_text)
{
    append_english(final_text, final_length);
    (void)strncpy(tentative, tentative_text, sizeof(tentative) - 1U);
    tentative[sizeof(tentative) - 1U] = '\0';
    ++revision;
}

void xl_display_poll(void)
{
    const uint32_t now = now_ms();
    int32_t x;
    int32_t y;

    if (!started)
    {
        if ((now - started_at) < XL_LOGO_MS)
        {
            return;
        }
        started = 1;
        start_speaker();
        touch_start();
        (void)read_press(now, &x, &y); /* take the inputs as they are: no press from before */
    }

    if (xl_router_poll(now))
    {
        ++revision;
    }

    /* Power state follows the source the router wants. */
    if (xl_router_target() == XL_SOURCE_N6)
    {
        if (state != STATE_TRANSLATING)
        {
            wake_translator();
        }
    }
    else if (state == STATE_TRANSLATING)
    {
        enter_music(now);
    }
    else if ((state == STATE_MUSIC_SCREEN) && ((int32_t)(now - screen_until) >= 0))
    {
        enter_sleep();
    }

    if (read_press(now, &x, &y))
    {
        if (state == STATE_MUSIC_SLEEP)
        {
            wake_screen(now); /* the tap that wakes the screen does nothing else */
        }
        else
        {
            if (state == STATE_MUSIC_SCREEN)
            {
                screen_until = now + XL_WAKE_SCREEN_MS;
            }
            handle_press(x, y, now);
        }
    }

    /* Two short beeps when the translator gets the speaker: the wiring works. */
    if (chime_pending && audio_ok && (xl_router_source() == XL_SOURCE_N6) && xl_audio_out_running())
    {
        (void)xl_audio_out_tone(660U, 120U, 12000);
        (void)xl_audio_out_tone(880U, 160U, 12000);
        chime_pending = 0;
    }

    if (state == STATE_MUSIC_SLEEP)
    {
        return;
    }
    if ((note != NULL) && ((int32_t)(now - note_until) >= 0))
    {
        ++revision; /* the note expired */
    }
    if ((revision != drawn_revision) || (drawn_mode != xl_router_mode()) || (drawn_music != xl_router_music()) ||
        (drawn_source != xl_router_target()))
    {
        draw_screen(now);
    }
}

ULONG xl_display_poll_ticks(void)
{
    const uint32_t ms = (state == STATE_MUSIC_SLEEP) ? XL_POLL_ASLEEP_MS : XL_POLL_AWAKE_MS;

    return (ULONG)(((ms * TX_TIMER_TICKS_PER_SECOND) + 999U) / 1000U);
}
