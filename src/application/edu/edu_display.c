/**
 * @file     edu_display.c
 * @brief    EDU mode screen; implements edu_display.h.
 * @details  Three screens on the 800x480 LCD:
 *           - ask:     score, "23 x 8 = ?", a blue SPEAK button;
 *           - listen:  the words heard so far (unconfirmed ones in grey), what they read as, a red
 *                      STOP button;
 *           - result:  a large green tick, a red cross, or an orange "?" (say it again), and why.
 *           Input: a tap on the button (GT911 touch screen), or the blue USER button, which works
 *           even when the touch screen does not answer.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*******************************************************************************/
/*                                   INCLUDES                                  */
/*******************************************************************************/

#include "edu_display.h"

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

#include "edu_game.h"
#include "logger.h"
#include "mw_lcd.h"
#include "stm32n6570_discovery.h"
#include "stm32n6570_discovery_bus.h"
#include "stm32n6570_discovery_ts.h"

/*******************************************************************************/
/*                               PRIVATE DEFINES                               */
/*******************************************************************************/

/** @brief How long the startup logo stays before the first question. */
#define EDU_LOGO_MS 1500U

/** @brief Touch screen start-up: attempts, and the wait between two. */
#define EDU_TOUCH_ATTEMPTS 5U
#define EDU_TOUCH_RETRY_MS 200U

/** @brief A second press within this time is ignored (bounce, a finger resting). */
#define EDU_PRESS_LOCKOUT_MS 300U

/** @brief While listening, redraw this often so the dots move. */
#define EDU_ANIMATION_MS 400U

/** @brief Layout (pixels). */
#define EDU_MARGIN        20
#define EDU_SCORE_Y       8
#define EDU_QUESTION_Y    80
#define EDU_LINE1_Y       220
#define EDU_LINE2_Y       285
#define EDU_BUTTON_W      300
#define EDU_BUTTON_H      100
#define EDU_BUTTON_Y      360
#define EDU_BUTTON_HIT    40   /* the button answers a little outside its edges */
#define EDU_SYMBOL_X      400
#define EDU_SYMBOL_Y      170
#define EDU_SYMBOL_SIZE   300
#define EDU_RESULT_LINE1_Y 300
#define EDU_RESULT_LINE2_Y 355
#define EDU_NEXT_Y        410
#define EDU_NEXT_H        64

/** @brief Room for one text line drawn on screen. */
#define EDU_TEXT_CAPACITY (EDU_GAME_HEARD_CAPACITY + EDU_GAME_TENTATIVE_CAPACITY)

/*******************************************************************************/
/*                              PRIVATE VARIABLES                              */
/*******************************************************************************/

/** @brief The quiz; static: it holds the checker's working memory (about 10 KB). */
static edu_game_t game;

/** @brief Scratch line for drawing. */
static char text[EDU_TEXT_CAPACITY];

static int touch_ready;
static int touch_down;
static int button_down;
static uint32_t last_press;
static uint32_t drawn_revision = 0xFFFFFFFFU;
static uint32_t drawn_at;
static uint32_t animation;
static edu_game_phase_t logged_phase;
static uint32_t started_at;
static int started;

static uint16_t color_blue;
static uint16_t color_red;
static uint16_t color_green;
static uint16_t color_orange;
static uint16_t color_grey;
static uint16_t color_light;

/*******************************************************************************/
/*                              PRIVATE FUNCTIONS                              */
/*******************************************************************************/

/** @brief Milliseconds since start (wraps after 49 days). */
static uint32_t now_ms(void)
{
#if (TX_TIMER_TICKS_PER_SECOND == 1000)
    return (uint32_t)tx_time_get();
#else
    return (uint32_t)(((uint64_t)tx_time_get() * 1000U) / TX_TIMER_TICKS_PER_SECOND);
#endif
}

static void put_text(char *line, uint32_t size, const char *s)
{
    uint32_t used = (uint32_t)strlen(line);

    while ((*s != '\0') && (used + 1U < size))
    {
        line[used++] = *s++;
    }
    line[used] = '\0';
}

static void put_number(char *line, uint32_t size, int32_t value)
{
    char digits[12];
    char s[12];
    uint32_t count = 0U;
    uint32_t k = 0U;

    do
    {
        digits[count++] = (char)('0' + (value % 10));
        value /= 10;
    } while ((value > 0) && (count < sizeof(digits) - 1U));
    while (count > 0U)
    {
        s[k++] = digits[--count];
    }
    s[k] = '\0';
    put_text(line, size, s);
}

static int32_t screen_width(void)
{
    return (int32_t)mw_lcd_get_width();
}

static void draw_at(int32_t x, int32_t y, const char *s, uint16_t scale, uint16_t color)
{
    mw_lcd_draw_string((uint16_t)((x < 0) ? 0 : x), (uint16_t)y, s, scale, color, true);
}

static void draw_centered(const char *s, int32_t y, uint16_t scale, uint16_t color)
{
    const int32_t width = (int32_t)mw_lcd_measure_string_width(s, scale);

    draw_at((screen_width() - width) / 2, y, s, scale, color);
}

/** @brief The button: a filled bar with its label in white. */
static void draw_button(int32_t y, int32_t height, const char *label, uint16_t color)
{
    const int32_t x = (screen_width() - EDU_BUTTON_W) / 2;
    const int32_t label_y = y + ((height - (int32_t)mw_lcd_get_line_height(1U)) / 2) + 4;

    mw_lcd_fill_rect(x, y, EDU_BUTTON_W, height, color);
    draw_centered(label, label_y, 1U, MW_LCD_COLOR_WHITE);
}

static void draw_score(void)
{
    text[0] = '\0';
    put_text(text, sizeof(text), "Right ");
    put_number(text, sizeof(text), game.right);
    put_text(text, sizeof(text), "   Wrong ");
    put_number(text, sizeof(text), game.wrong);
    put_text(text, sizeof(text), "   Streak ");
    put_number(text, sizeof(text), game.streak);
    draw_at(EDU_MARGIN, EDU_SCORE_Y, text, 1U, color_grey);
    mw_lcd_fill_rect(0, EDU_SCORE_Y + (int32_t)mw_lcd_get_line_height(1U), screen_width(), 2, color_light);
}

static void draw_question(void)
{
    text[0] = '\0';
    put_number(text, sizeof(text), game.question[0]);
    put_text(text, sizeof(text), " x ");
    put_number(text, sizeof(text), game.question[1]);
    put_text(text, sizeof(text), " = ?");
    draw_centered(text, EDU_QUESTION_Y, 2U, MW_LCD_COLOR_BLACK);
}

/** @brief The words heard: committed in black, then the tentative tail in grey; the newest part that fits. */
static void draw_heard(int32_t y)
{
    const int32_t room = screen_width() - (2 * EDU_MARGIN);
    const uint32_t committed = (uint32_t)strlen(game.heard);
    uint32_t total;
    uint32_t start = 0U;
    int32_t x = EDU_MARGIN;

    text[0] = '\0';
    put_text(text, sizeof(text), game.heard);
    put_text(text, sizeof(text), game.tentative);
    total = (uint32_t)strlen(text);
    while ((start < total) && ((int32_t)mw_lcd_measure_string_width(&text[start], 1U) > room))
    {
        ++start; /* drop the oldest characters until the line fits */
    }
    if (start < committed)
    {
        const char saved = text[committed];

        text[committed] = '\0';
        draw_at(x, y, &text[start], 1U, MW_LCD_COLOR_BLACK);
        x += (int32_t)mw_lcd_measure_string_width(&text[start], 1U);
        text[committed] = saved;
        start = committed;
    }
    draw_at(x, y, &text[start], 1U, color_grey);
}

static void draw_tick(int32_t cx, int32_t cy, int32_t s, uint16_t color)
{
    const uint16_t thick = (uint16_t)((s * 13) / 100);

    mw_lcd_draw_line(cx - (s * 36) / 100, cy + (s * 2) / 100, cx - (s * 12) / 100, cy + (s * 28) / 100, thick, color);
    mw_lcd_draw_line(cx - (s * 12) / 100, cy + (s * 28) / 100, cx + (s * 38) / 100, cy - (s * 30) / 100, thick, color);
}

static void draw_cross(int32_t cx, int32_t cy, int32_t s, uint16_t color)
{
    const uint16_t thick = (uint16_t)((s * 13) / 100);
    const int32_t r = (s * 30) / 100;

    mw_lcd_draw_line(cx - r, cy - r, cx + r, cy + r, thick, color);
    mw_lcd_draw_line(cx + r, cy - r, cx - r, cy + r, thick, color);
}

static void draw_screen(uint32_t now)
{
    mw_lcd_clear(MW_LCD_COLOR_WHITE);
    draw_score();
    switch (game.phase)
    {
    case EDU_GAME_ASK:
        draw_question();
        draw_centered(touch_ready ? "Tap SPEAK and answer" : "Press USER and answer", EDU_LINE1_Y, 1U, color_grey);
        draw_button(EDU_BUTTON_Y, EDU_BUTTON_H, "SPEAK", color_blue);
        break;
    case EDU_GAME_LISTEN:
    {
        static const char *const dots[4] = { "Listening", "Listening.", "Listening..", "Listening..." };

        draw_question();
        if ((game.heard[0] != '\0') || (game.tentative[0] != '\0'))
        {
            draw_heard(EDU_LINE1_Y);
        }
        else
        {
            draw_centered(dots[animation % 4U], EDU_LINE1_Y, 1U, color_grey);
        }
        if (game.live[0] != '\0')
        {
            draw_centered(game.live, EDU_LINE2_Y, 1U, color_blue);
        }
        draw_button(EDU_BUTTON_Y, EDU_BUTTON_H, game.stopping ? "WAIT" : "STOP",
                    game.stopping ? color_grey : color_red);
        break;
    }
    default:
        if (game.verdict == EDU_GAME_RIGHT)
        {
            draw_tick(EDU_SYMBOL_X, EDU_SYMBOL_Y, EDU_SYMBOL_SIZE, color_green);
        }
        else if (game.verdict == EDU_GAME_WRONG)
        {
            draw_cross(EDU_SYMBOL_X, EDU_SYMBOL_Y, EDU_SYMBOL_SIZE, color_red);
        }
        else
        {
            const int32_t width = (int32_t)mw_lcd_measure_string_width("?", 4U);

            draw_at(EDU_SYMBOL_X - (width / 2), EDU_SYMBOL_Y - 130, "?", 4U, color_orange);
        }
        draw_centered(game.line1, EDU_RESULT_LINE1_Y, 1U, MW_LCD_COLOR_BLACK);
        draw_centered(game.line2, EDU_RESULT_LINE2_Y, 1U,
                      (game.verdict == EDU_GAME_WRONG) ? color_red :
                      ((game.verdict == EDU_GAME_RIGHT) ? color_green : color_grey));
        draw_button(EDU_NEXT_Y, EDU_NEXT_H, "NEXT", color_grey);
        break;
    }
    if (mw_lcd_present() != MW_LCD_OK)
    {
        LOG_ERROR("EDU: LCD presentation failed");
    }
    drawn_revision = game.revision;
    drawn_at = now;
}

/** @brief A tap counts when it lands on (or near) the button of the screen shown. */
static int on_button(uint32_t x, uint32_t y)
{
    const int32_t left = ((screen_width() - EDU_BUTTON_W) / 2) - EDU_BUTTON_HIT;
    const int32_t right = left + EDU_BUTTON_W + (2 * EDU_BUTTON_HIT);
    const int32_t top = ((game.phase == EDU_GAME_RESULT) ? EDU_NEXT_Y : EDU_BUTTON_Y) - EDU_BUTTON_HIT;

    return ((int32_t)x >= left) && ((int32_t)x <= right) && ((int32_t)y >= top);
}

/** @brief Poll the touch screen and the USER button; 1 on a new press. */
static int pressed(uint32_t now)
{
    int press = 0;
    const int button = (BSP_PB_GetState(BUTTON_USER1) != 0U);

    if (button && !button_down)
    {
        press = 1;
    }
    button_down = button;
    if (touch_ready)
    {
        TS_State_t state;

        if (BSP_TS_GetState(0U, &state) == BSP_ERROR_NONE)
        {
            const int down = (state.TouchDetected != 0U);

            if (down && !touch_down && on_button(state.TouchX, state.TouchY))
            {
                press = 1;
            }
            touch_down = down;
        }
    }
    if (press && ((now - last_press) < EDU_PRESS_LOCKOUT_MS))
    {
        press = 0;
    }
    if (press)
    {
        last_press = now;
    }
    return press;
}

/**
 * @brief Bring up the touch screen (GT911 on I2C2), with retries; log why if it does not answer.
 * @note The GT911 comes out of reset with the LCD (shared NRST) and boots for tens of
 *       milliseconds; ST's driver probes it at once, so this runs after the logo and retries.
 */
static void touch_start(void)
{
    TS_Init_t ts;
    int32_t status = BSP_ERROR_NO_INIT;
    uint32_t attempt;

    ts.Width = mw_lcd_get_width();
    ts.Height = mw_lcd_get_height();
    ts.Orientation = TS_SWAP_NONE;
    ts.Accuracy = 0U;
    for (attempt = 1U; attempt <= EDU_TOUCH_ATTEMPTS; ++attempt)
    {
        status = BSP_TS_Init(0U, &ts);
        if (status == BSP_ERROR_NONE)
        {
            break;
        }
        (void)tx_thread_sleep((EDU_TOUCH_RETRY_MS * TX_TIMER_TICKS_PER_SECOND) / 1000U);
    }
    touch_ready = (status == BSP_ERROR_NONE);
    if (touch_ready)
    {
        LOG_INFO("EDU: times-table quiz; tap SPEAK on the screen or press USER (touch ready, attempt %lu)",
                 (unsigned long)attempt);
        return;
    }
    {
        /* Which addresses answer on I2C2: the GT911 picks 0xBA or 0x28 when it leaves reset. */
        const int32_t bus = BSP_I2C2_Init();
        const int32_t at_ba = (bus == BSP_ERROR_NONE) ? BSP_I2C2_IsReady(0xBAU, 3U) : bus;
        const int32_t at_28 = (bus == BSP_ERROR_NONE) ? BSP_I2C2_IsReady(0x28U, 3U) : bus;

        LOG_WARNING("EDU: touch screen not found (BSP_TS_Init %ld after %lu tries; I2C2 init %ld, "
                    "0xBA %s, 0x28 %s); press the USER button to speak",
                    (long)status, (unsigned long)EDU_TOUCH_ATTEMPTS, (long)bus,
                    (at_ba == BSP_ERROR_NONE) ? "answers" : "silent", (at_28 == BSP_ERROR_NONE) ? "answers" : "silent");
    }
}

/** @brief One line on the console for each verdict, to compare with the PC game's logs. */
static void log_turn(void)
{
    if ((game.phase == EDU_GAME_RESULT) && (logged_phase != EDU_GAME_RESULT))
    {
        static const char *const names[] = { "-", "right", "wrong", "unclear", "nothing" };

        LOG_INFO("EDU %d x %d: heard \"%s\" -> %s (%s)", game.question[0], game.question[1], game.heard,
                 names[game.verdict], game.line1);
    }
    logged_phase = game.phase;
}

/*******************************************************************************/
/*                               PUBLIC FUNCTIONS                              */
/*******************************************************************************/

void edu_display_start(void)
{
    color_blue = mw_lcd_rgb888_to_rgb565(0x1EU, 0x6FU, 0xD9U);
    color_red = mw_lcd_rgb888_to_rgb565(0xD9U, 0x30U, 0x25U);
    color_green = mw_lcd_rgb888_to_rgb565(0x1EU, 0x9EU, 0x3EU);
    color_orange = mw_lcd_rgb888_to_rgb565(0xE0U, 0x8AU, 0x00U);
    color_grey = mw_lcd_rgb888_to_rgb565(0x80U, 0x80U, 0x80U);
    color_light = mw_lcd_rgb888_to_rgb565(0xE0U, 0xE0U, 0xE0U);

    if (BSP_PB_Init(BUTTON_USER1, BUTTON_MODE_GPIO) != BSP_ERROR_NONE)
    {
        LOG_WARNING("EDU: USER button not available");
    }

    /* The touch screen starts when the logo goes (touch_start()): its controller shares the
       LCD's reset, released just before, and needs time to boot. */
    edu_game_init(&game, (uint32_t)tx_time_get() ^ 0x5EEDU);
    logged_phase = game.phase;
    started_at = now_ms();
}

void edu_display_text(const char *final_text, uint32_t final_length, const char *tentative)
{
    edu_game_text(&game, final_text, final_length, tentative, now_ms());
}

void edu_display_poll(void)
{
    const uint32_t now = now_ms();

    if (!started)
    {
        if ((now - started_at) < EDU_LOGO_MS)
        {
            return;
        }
        started = 1;
        touch_start();
        (void)pressed(now); /* take the button state as it is now: no press from before */
        draw_screen(now);
        return;
    }
    if (pressed(now))
    {
        edu_game_press(&game, now);
    }
    edu_game_tick(&game, now);
    log_turn();
    if (game.phase == EDU_GAME_LISTEN)
    {
        if ((now - drawn_at) >= EDU_ANIMATION_MS)
        {
            ++animation;
            drawn_revision = game.revision - 1U; /* force a redraw */
        }
    }
    if (game.revision != drawn_revision)
    {
        draw_screen(now);
    }
}
