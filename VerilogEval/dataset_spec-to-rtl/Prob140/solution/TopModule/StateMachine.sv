module StateMachine (
    input  wire       clk,
    input  wire       reset,
    input  wire       in,
    output reg [2:0]  state
);

    // State encoding
    typedef enum logic [2:0] {
        IDLE    = 3'd0,
        S1      = 3'd1,
        S2      = 3'd2,
        S3      = 3'd3,
        S4      = 3'd4,
        S5      = 3'd5,
        S6      = 3'd6,
        ERROR   = 3'd7
    } state_t;

    state_t current_state, next_state;

    // State register
    always_ff @(posedge clk) begin
        if (reset)
            current_state <= IDLE;
        else
            current_state <= next_state;
    end

    // Next state logic
    always_comb begin
        case (current_state)
            IDLE: begin
                if (in)
                    next_state = S1;
                else
                    next_state = IDLE;
            end
            S1: begin
                if (in)
                    next_state = S2;
                else
                    next_state = IDLE;
            end
            S2: begin
                if (in)
                    next_state = S3;
                else
                    next_state = IDLE;
            end
            S3: begin
                if (in)
                    next_state = S4;
                else
                    next_state = IDLE;
            end
            S4: begin
                if (in)
                    next_state = S5;
                else
                    next_state = IDLE;
            end
            S5: begin
                if (in)
                    next_state = S6;
                else
                    next_state = IDLE;
            end
            S6: begin
                if (in)
                    next_state = ERROR;
                else
                    next_state = IDLE;
            end
            ERROR: begin
                next_state = ERROR; // Remain in error state
            end
            default: next_state = IDLE;
        endcase
    end

    // Output logic (Moore machine)
    always_comb begin
        state = current_state;
    end

endmodule