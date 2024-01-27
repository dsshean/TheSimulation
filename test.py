import curses
import random

def main(stdscr):
    # Initialize colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Trees
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Player
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Roads

    # Set up the window
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)

    # Get the size of the terminal
    max_y, max_x = stdscr.getmaxyx()
    grid_size = min(max_x - 1, max_y - 1, 100)

    # Game variables
    pos_x, pos_y = grid_size // 2, grid_size // 2
    player_char = u"\u263A"  # Unicode character for the player
    tree_char = u"\u2663"    # Unicode character for trees (♣)

    # Initialize the forest
    forest = [[tree_char if random.random() < 0.1 else '.' for _ in range(grid_size)] for _ in range(grid_size)]

    while True:
        # Draw the grid
        stdscr.clear()
        for y in range(grid_size):
            for x in range(grid_size):
                if x == pos_x and y == pos_y:
                    stdscr.addch(y, x, player_char, curses.color_pair(2))
                else:
                    char = forest[y][x]
                    color = curses.color_pair(1) if char == tree_char else curses.color_pair(3)
                    stdscr.addch(y, x, char, color)

        # Handle input
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == ord('w') and pos_y > 0:
            pos_y -= 1
        elif key == ord('s') and pos_y < grid_size - 2:
            pos_y += 1
        elif key == ord('a') and pos_x > 0:
            pos_x -= 1
        elif key == ord('d') and pos_x < grid_size - 2:
            pos_x += 1

        # Refresh the screen
        stdscr.refresh()

if __name__ == "__main__":
    curses.wrapper(main)
