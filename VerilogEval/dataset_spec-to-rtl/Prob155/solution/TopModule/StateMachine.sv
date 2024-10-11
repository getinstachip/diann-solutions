module StateMachine (
    input  logic       clk,
    input  logic       areset,
    input  logic       bump_left,
    input  logic       bump_right,
    input  logic       ground,
    input  logic       dig,
    output logic [2:0] state
);

    // State encoding
    typedef enum logic [2:0] {
        WALK_LEFT   = 3'b000,
        WALK_RIGHT  = 3'b001,
        FALLING     = 3'b010,
        DIGGING     = 3'b011,
        SPLATTERED  = 3'b100
    } state_t;

    state_t current_state, next_state;

    // Counter for falling cycles
    logic [5:0] fall_counter; // 6 bits to count up to 63

    // Register to hold previous walking direction before falling/digging
    logic prev_walk_left;

    // State and counter register
    always_ff @(posedge clk or posedge areset) begin
        if (areset) begin
            current_state <= WALK_LEFT;
            fall_counter <= 6'd0;
            prev_walk_left <= 1'b1;
        end
        else begin
            current_state <= next_state;
            if (current_state == FALLING) begin
                if (ground)
                    fall_counter <= 6'd0;
                else
                    fall_counter <= fall_counter + 1;
            end
            else begin
                fall_counter <= 6'd0;
            end

            if (current_state == WALK_LEFT)
                prev_walk_left <= 1'b1;
            else if (current_state == WALK_RIGHT)
                prev_walk_left <= 1'b0;
        end
    end

    // Next state logic
    always_comb begin
        next_state = current_state; // default

        if (current_state != SPLATTERED) begin
            case (current_state)
                WALK_LEFT: begin
                    if (!ground) begin
                        next_state = FALLING;
                    end
                    else if (dig) begin
                        next_state = DIGGING;
                    end
                    else if (bump_left || bump_right) begin
                        next_state = WALK_RIGHT;
                    end
                end

                WALK_RIGHT: begin
                    if (!ground) begin
                        next_state = FALLING;
                    end
                    else if (dig) begin
                        next_state = DIGGING;
                    end
                    else if (bump_left || bump_right) begin
                        next_state = WALK_LEFT;
                    end
                end

                FALLING: begin
                    if (ground) begin
                        if (fall_counter + 1 > 20)
                            next_state = SPLATTERED;
                        else if (prev_walk_left)
                            next_state = WALK_LEFT;
                        else
                            next_state = WALK_RIGHT;
                    end
                end

                DIGGING: begin
                    if (!ground) begin
                        next_state = FALLING;
                    end
                end

                default: next_state = SPLATTERED;
            endcase
        end
    end

    // Output assignment
    assign state = current_state;

endmodule