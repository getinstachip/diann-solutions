module TimerCounter (
    input  wire        clk,
    input  wire        reset,
    input  wire        start_count,
    input  wire [3:0]  delay,
    output reg  [3:0]  count,
    output reg         counting,
    output reg         done,
    output reg  [3:0]  remaining_time
);

    // Parameters
    localparam CYCLE_MAX = 1000;

    // State Encoding
    typedef enum logic [1:0] {
        IDLE    = 2'b00,
        COUNT   = 2'b01,
        DONE_STATE = 2'b10
    } state_t;

    state_t current_state, next_state;

    // Counters
    reg [9:0] cycle_counter;      // To count up to 1000
    reg [3:0] delay_counter;      // To count delay units

    // State Transition
    always_ff @(posedge clk) begin
        if (reset) begin
            current_state <= IDLE;
        end else begin
            current_state <= next_state;
        end
    end

    // Next State Logic
    always_comb begin
        case (current_state)
            IDLE: begin
                if (start_count)
                    next_state = COUNT;
                else
                    next_state = IDLE;
            end
            COUNT: begin
                if (delay_counter == 4'd0 && cycle_counter == (CYCLE_MAX-1))
                    next_state = DONE_STATE;
                else
                    next_state = COUNT;
            end
            DONE_STATE: begin
                next_state = DONE_STATE; // Remain until externally reset
            end
            default: next_state = IDLE;
        endcase
    end

    // Sequential Logic
    always_ff @(posedge clk) begin
        if (reset) begin
            counting       <= 1'b0;
            done           <= 1'b0;
            cycle_counter  <= 10'd0;
            delay_counter  <= 4'd0;
            count          <= 4'd0;
            remaining_time <= 4'd0;
        end else begin
            case (current_state)
                IDLE: begin
                    if (start_count) begin
                        delay_counter  <= delay;
                        remaining_time <= delay;
                        counting       <= 1'b1;
                        done           <= 1'b0;
                        cycle_counter  <= 10'd0;
                        count          <= delay;
                    end
                end
                COUNT: begin
                    if (cycle_counter < (CYCLE_MAX-1)) begin
                        cycle_counter <= cycle_counter + 1;
                    end else begin
                        cycle_counter <= 10'd0;
                        if (delay_counter > 4'd0) begin
                            delay_counter  <= delay_counter - 1;
                            remaining_time <= delay_counter - 1;
                            count          <= delay_counter - 1;
                        end
                    end

                    if (delay_counter == 4'd0 && cycle_counter == (CYCLE_MAX-1)) begin
                        counting <= 1'b0;
                        done     <= 1'b1;
                    end
                end
                DONE_STATE: begin
                    // Remain in DONE_STATE until reset externally
                    // All outputs are already set
                end
                default: begin
                    // Default case
                end
            endcase
        end
    end

endmodule